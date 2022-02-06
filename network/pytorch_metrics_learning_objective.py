import torch
from torch import nn
from torch.nn import functional as F
from pytorch_metric_learning.miners import MultiSimilarityMiner, TripletMarginMiner, BatchHardMiner
from pytorch_metric_learning.losses import TripletMarginLoss, CircleLoss, ArcFaceLoss

custom_namespace = {
    'MultiSimilarityMiner': MultiSimilarityMiner,
    'TripletMarginLoss': TripletMarginLoss,
    'TripletMarginMiner': TripletMarginMiner,
    'BatchHardMiner': BatchHardMiner,
    'CircleLoss': CircleLoss,
    'ArcFaceLoss': ArcFaceLoss
}


class PytorchMetricLearningObjectiveWithSampling(nn.Module):
    def __init__(
            self,
            lambdas=8e-5,
            top_anomaly_frames=3,
            top_normal_frames=3,
            loss_name="TripletMarginLoss",
            miner_name="MultiSimilarityMiner",
    ):
        super(PytorchMetricLearningObjectiveWithSampling, self).__init__()
        self.lambdas = lambdas
        self.top_anomaly_frames = top_anomaly_frames
        self.top_normal_frames = top_normal_frames
        if loss_name == 'ArcFaceLoss':
            self.loss_func = custom_namespace[loss_name](num_classes=2, embedding_size=128).to('cuda:0')
            print('W shape:', self.loss_func.W.shape)
        else:
            self.loss_func = custom_namespace[loss_name]()
        self.miner = custom_namespace[miner_name]()

    def forward(self, normalized_embeddings, y_true):
        bs, segm_len, embed_dim = normalized_embeddings.shape
        normal_vids_indices = torch.where(y_true == 0)
        anomal_vids_indices = torch.where(y_true == 1)

        normal_segments_embeddings = normalized_embeddings[normal_vids_indices].squeeze(
            -1
        )  # (batch/2, segm_len, embed_dim)
        anomal_segments_embeddings = normalized_embeddings[anomal_vids_indices].squeeze(
            -1
        )  # (batch/2, segm_len, embed_dim)

        # anchors are the frame 0 of the anomaly video
        anchors = anomal_segments_embeddings[:, :1, :]  # (batch/2, 1, embed_dim)

        """
        Prepare positive (anomaly) frames embeddings
        To do that we pick N=top_anomaly_frames frames from anomaly video that are distant from the corresponding anchor
        """
        n_anom_video, n_frames, embed_dim = anomal_segments_embeddings.shape
        d_to_anomalies = (
            (anomal_segments_embeddings - anchors).pow(2).sum(-1)
        )  # (batch/2, segm_len, embed_dim)
        tp_frame_inds = torch.argsort(d_to_anomalies, dim=1)[
                        :, -self.top_anomaly_frames:
                        ]  # (batch/2,top_anomaly_frames)

        tp_frame_mask = torch.zeros((n_anom_video, n_frames), dtype=torch.bool).to(
            anomal_segments_embeddings.device
        )
        for row_idx in range(tp_frame_inds.shape[0]):
            for col_idx in tp_frame_inds[row_idx]:
                tp_frame_mask[row_idx, col_idx] = True

        positive_frames = anomal_segments_embeddings[tp_frame_mask].view(
            n_anom_video, -1, embed_dim
        )  # (batch/2,top_anomaly_frames, embed_dim)

        """
        Prepare negative (normal) frames embeddings
        To do that we pick N=top_normal_frames frames from normal videos that are distant from the anchors
        Here we calculate pairwise distance to pick hard negative examples for each anchor
        """
        n_norm_video, _, _ = normal_segments_embeddings.shape
        fp_frame_inds = torch.cdist(normal_segments_embeddings, anchors).argsort(dim=1)[
                        :, -self.top_normal_frames:, 0
                        ]
        fp_frame_mask = torch.zeros((n_norm_video, n_frames), dtype=torch.bool).to(
            anomal_segments_embeddings.device
        )
        for row_idx in range(fp_frame_inds.shape[0]):
            for col_idx in fp_frame_inds[row_idx]:
                fp_frame_mask[row_idx, col_idx] = True

        negative_frames = normal_segments_embeddings[fp_frame_mask].view(
            n_norm_video, -1, embed_dim
        )  # (batch/2,top_normal_frames, embed_dim)

        """
        Metric learning loss
        """
        embeddings = torch.cat(
            [
                positive_frames.reshape(-1, embed_dim),
                negative_frames.reshape(-1, embed_dim),
            ]
        )  # (bs*top_anomaly_frames + bs*top_normal_frames, embed_dim)
        labels = torch.Tensor(
            [1] * (bs // 2 * self.top_anomaly_frames)
            + [0] * (bs // 2 * self.top_normal_frames)).to(embeddings.device).type(torch.long)

        hard_pairs = self.miner(embeddings, labels)
        triplet_loss = self.loss_func(embeddings, labels, hard_pairs)
        """
        Smoothness of anomalous video
        """
        # smoothed_scores = (
        #         anomal_segments_embeddings[:, 1:] - anomal_segments_embeddings[:, :-1]
        # )
        # smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)
        smoothed_scores_distance = self.loss_func.distance(
            anomal_segments_embeddings[:, 1:].reshape(-1, embed_dim),
            anomal_segments_embeddings[:, :-1].reshape(-1, embed_dim)
        )
        if self.loss_func.distance.is_inverted:
            smoothed_scores_distance = -smoothed_scores_distance
        final_loss = (triplet_loss + self.lambdas * smoothed_scores_distance).mean()
        return final_loss
