import argparse
import os
import numpy as np
import random
from datetime import datetime
import uuid
import json
from os import path
import inspect
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from features_loader import FeaturesLoader
from network.TorchUtils import TorchModel
from network.anomaly_detector_model import (
    AnomalyDetector,
    custom_objective,
    RegularizedLoss,
    TripletRegularizedLoss,
)
from network.pytorch_metrics_learning_objective import (
    PytorchMetricLearningObjectiveWithSampling,
)
from network.triplet_anomaly_detector_model import TripletAnomalyDetector
from network.triplet_loss import triplet_objective, triplet_objective_sampling
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger, get_torch_device

custom_namespace = {
    "TripletAnomalyDetector": TripletAnomalyDetector,
    "AnomalyDetector": AnomalyDetector,
    "custom_objective": custom_objective,
    "triplet_objective": triplet_objective,
    "triplet_objective_sampling": triplet_objective_sampling,
    "PytorchMetricLearningObjectiveWithSampling": PytorchMetricLearningObjectiveWithSampling,
}


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

    # io
    parser.add_argument("--features_path", default="features", help="path to features")
    parser.add_argument(
        "--annotation_path",
        default="Train_Annotation.txt",
        help="path to train annotation",
    )
    parser.add_argument(
        "--log_file", type=str, default="log.log", help="set logging file."
    )
    parser.add_argument(
        "--exps_dir",
        type=str,
        default="exps",
        help="path to the directory where models and tensorboard would be saved.",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="load a model for resume training"
    )

    # optimization
    parser.add_argument("--batch_size", type=int, default=60, help="batch size")
    # model params
    parser.add_argument(
        "--network_name", type=str, default="AnomalyDetector", help="name of network"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=4096, help="feature dimensionality"
    )
    use_last_bn_parser = parser.add_mutually_exclusive_group(required=False)
    use_last_bn_parser.add_argument(
        "--use_last_bn", dest="use_last_bn", action="store_true"
    )
    use_last_bn_parser.add_argument(
        "--no_use_last_bn", dest="use_last_bn", action="store_false"
    )
    use_last_bn_parser.set_defaults(use_last_bn=True)

    norm_out_to_unit_parser = parser.add_mutually_exclusive_group(required=False)
    norm_out_to_unit_parser.add_argument(
        "--norm_out_to_unit", dest="norm_out_to_unit", action="store_true"
    )
    norm_out_to_unit_parser.add_argument(
        "--no_norm_out_to_unit", dest="norm_out_to_unit", action="store_false"
    )
    norm_out_to_unit_parser.set_defaults(norm_out_to_unit=True)

    norm_on_eval_parser = parser.add_mutually_exclusive_group(required=False)
    norm_on_eval_parser.add_argument(
        "--norm_on_eval", dest="norm_on_eval", action="store_true"
    )
    norm_on_eval_parser.add_argument(
        "--no_norm_on_eval", dest="norm_on_eval", action="store_false"
    )
    norm_on_eval_parser.set_defaults(norm_on_eval=True)

    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="epochs interval for saving the model checkpoints",
    )
    parser.add_argument("--optimizer", type=str, default="adadelta", help="optimizer")
    parser.add_argument("--lr_base", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--iterations_per_epoch",
        type=int,
        default=2000,
        help="number of training iterations",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )
    parser.add_argument(
        "--objective_name",
        type=str,
        default="custom_objective",
        help="name of objective function",
    )

    parser.add_argument(
        "--output_dim",
        type=int,
        default=128,
        help="dimension of output of network for triplet loss",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="dropout rate in network for triplet loss",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        default=8e-5,
        help="lambdas value for smoothness part in triplet loss",
    )
    parser.add_argument(
        "--top_anomaly_frames",
        type=int,
        default=3,
        help="top anomalaly frames (segments) per video",
    )
    parser.add_argument(
        "--top_normal_frames",
        type=int,
        default=3,
        help="top normal frames (segments) per video",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2, help="margin constant in triplet loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--loss_name",
        type=str,
        default="TripletMarginLoss",
        help="loss name from pytorch metrics learning",
    )
    parser.add_argument(
        "--miner_name",
        type=str,
        default="MultiSimilarityMiner",
        help="miner name from pytorch metrics learning",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Register directories
    register_logger(log_file=args.log_file)
    os.makedirs(args.exps_dir, exist_ok=True)

    experiment_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_unique_hash = str(uuid.uuid4())[:7]
    exp_name = "_".join(
        [
            args.network_name,
            args.objective_name,
            experiment_timestamp,
            experiment_unique_hash,
        ]
    )
    models_dir = path.join(args.exps_dir, exp_name)
    tb_dir = models_dir
    os.makedirs(models_dir, exist_ok=True)
    with open(path.join(models_dir, "params.json"), "w") as fp:
        json.dump(vars(args), fp, indent=4)

    # Optimizations
    device = get_torch_device()
    cudnn.benchmark = True  # enable cudnn tune
    set_seed(args.seed)

    # Data loader
    train_loader = FeaturesLoader(
        features_path=args.features_path,
        annotation_path=args.annotation_path,
        iterations=args.iterations_per_epoch,
    )

    # Model
    if args.checkpoint is not None and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
    else:
        if args.network_name == "TripletAnomalyDetector":
            network_params = {
                "input_dim": args.feature_dim,
                "output_dim": args.output_dim,
                "dropout_rate": args.dropout_rate,
                "use_last_bn": args.use_last_bn,
                "norm_out_to_unit": args.norm_out_to_unit,
                "norm_on_eval": args.norm_on_eval,
            }
            network = custom_namespace[args.network_name](**network_params)
        else:
            network = custom_namespace[args.network_name](args.feature_dim)
        model = TorchModel(network)

    model = model.to(device).train()
    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """


    if inspect.isfunction(custom_namespace[args.objective_name]):
        objective = custom_namespace[args.objective_name]
    elif inspect.isclass(custom_namespace[args.objective_name]):
        obj_params = {}
        if args.objective_name == "PytorchMetricLearningObjectiveWithSampling":
            obj_params = {
                "lambdas": args.lambdas,
                "top_anomaly_frames": args.top_anomaly_frames,
                "top_normal_frames": args.top_normal_frames,
                "loss_name": args.loss_name,
                "miner_name": args.miner_name,
            }
        objective = custom_namespace[args.objective_name](**obj_params)
    if args.objective_name == "triplet_objective_sampling":
        loss_params = {
            "triplet_lambdas": args.lambdas,
            "top_anomaly_frames": args.top_anomaly_frames,
            "top_normal_frames": args.top_normal_frames,
            "margin": args.margin,
        }
        criterion = TripletRegularizedLoss(
            network, objective, **loss_params
        ).to(device)
    else:
        criterion = RegularizedLoss(network, objective).to(device)

    if args.loss_name == 'ArcFace':
        opt_params = list(model.parameters()) + list(objective.loss_func.parameters())
    else:
        opt_params = model.parameters()

    if args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(opt_params, lr=args.lr_base, eps=1e-8)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(opt_params, lr=args.lr_base)

    # Callbacks
    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(visualization_dir=args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Training
    model.fit(
        train_iter=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        network_model_path_base=models_dir,
        save_every=args.save_every,
    )
