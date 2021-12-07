import argparse
import os
from os import path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from features_loader import FeaturesLoader
from network.TorchUtils import TorchModel
from network.anomaly_detector_model import AnomalyDetector, custom_objective, RegularizedLoss, TripletRegularizedLoss
from network.triplet_anomaly_detector_model import TripletAnomalyDetector
from network.triplet_loss import triplet_objective, triplet_objective_sampling
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger, get_torch_device

custom_namespace = {
    'TripletAnomalyDetector': TripletAnomalyDetector,
    'AnomalyDetector': AnomalyDetector,
    'custom_objective': custom_objective,
    'triplet_objective': triplet_objective,
    'triplet_objective_sampling': triplet_objective_sampling,
}

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

    # io
    parser.add_argument('--features_path', default='features',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                        help="path to train annotation")
    parser.add_argument('--log_file', type=str, default="log.log",
                        help="set logging file.")
    parser.add_argument('--exps_dir', type=str, default="exps",
                        help="path to the directory where models and tensorboard would be saved.")
    parser.add_argument('--checkpoint', type=str,
                        help="load a model for resume training")

    # optimization
    parser.add_argument('--batch_size', type=int, default=60,
                        help="batch size")
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help="batch size")
    parser.add_argument('--save_every', type=int, default=1,
                        help="epochs interval for saving the model checkpoints")
    parser.add_argument('--lr_base', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--iterations_per_epoch', type=int, default=20000,
                        help="number of training iterations")
    parser.add_argument('--epochs', type=int, default=2,
                        help="number of training epochs")
    parser.add_argument('--network_name', type=str, default='AnomalyDetector',
                        help="name of network")
    parser.add_argument('--objective_name', type=str, default='custom_objective',
                        help="name of objective function")
    
    parser.add_argument('--output_dim', type=int, default=128,
                        help="dimension of output of network for triplet loss")
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help="dropout rate in network for triplet loss")
    parser.add_argument('--lambdas', type=float, default=8e-5,
                        help="lambdas value for smoothness part in triplet loss")
    parser.add_argument('--top_anomaly_frames', type=int, default=3,
                        help="number of anomaly segments in triplet loss")
    parser.add_argument('--top_normal_frames', type=int, default=3,
                        help="number of normal segments in triplet loss")
    parser.add_argument('--margin', type=float, default=0.2,
                        help="margin constant in triplet lossn")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Register directories
    register_logger(log_file=args.log_file)
    os.makedirs(args.exps_dir, exist_ok=True)
    models_dir = path.join(args.exps_dir, 'models')
    tb_dir = path.join(args.exps_dir, 'tensorboard')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Optimizations
    device = get_torch_device()
    cudnn.benchmark = True  # enable cudnn tune

    # Data loader
    train_loader = FeaturesLoader(features_path=args.features_path,
                                  annotation_path=args.annotation_path,
                                  iterations=args.iterations_per_epoch)

    # Model
    if args.checkpoint is not None and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
    else:
        if 'Triplet' in args.network_name:
            network = custom_namespace[args.network_name](input_dim=args.feature_dim, 
                                                          output_dim=args.output_dim, 
                                                          dropout_rate=args.dropout_rate)
        else:
            network = custom_namespace[args.network_name](input_dim=args.feature_dim)
        model = TorchModel(network)

    model = model.to(device).train()
    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr_base, eps=1e-8)

    if 'triplet' in args.objective_name:
        criterion = TripletRegularizedLoss(network, 
                                           custom_namespace[args.objective_name], 
                                           triplet_lambdas=args.lambdas, 
                                           top_anomaly_frames=args.top_anomaly_frames,
                                           top_normal_frames=args.top_normal_frames, 
                                           margin=args.margin).to(device)
    else:
        criterion = RegularizedLoss(network, custom_namespace[args.objective_name]).to(device)

    # Callbacks
    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(visualization_dir=args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Training
    model.fit(train_iter=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              epochs=args.epochs,
              network_model_path_base=models_dir,
              save_every=args.save_every)
