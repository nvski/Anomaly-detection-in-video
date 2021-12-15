import torch
from torch.nn import functional as F
from pytorch_metric_learning.miners import MultiSimilarityMiner, TripletMarginMiner
from pytorch_metric_learning.losses import TripletMarginLoss


def triplet_objective(normalized_embeddings, y_true):
    # normalized_embeddings (batch_size, 32, embed_dim)
    # y_true (batch_size)
    lambdas = 8e-5
    top_anomaly_frames = 3
    top_normal_frames = 3
    margin = 0.2

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    normal_segments_embeddings = normalized_embeddings[normal_vids_indices].squeeze(
        -1
    )  # (batch/2, 32, embed_dim)
    anomal_segments_embeddings = normalized_embeddings[anomal_vids_indices].squeeze(
        -1
    )  # (batch/2, 32, embed_dim)

    # anchors are the frame 0 of the anomaly video
    anchors = anomal_segments_embeddings[:, :1, :]  # (batch/2, 1, embed_dim)

    """
    Prepare positive (anomaly) frames embeddings
    To do that we pick N=top_anomaly_frames frames from anomaly video that are distant from the corresponding anchor
    """
    n_anom_video, n_frames, embed_dim = anomal_segments_embeddings.shape
    d_to_anomalies = (
        (anomal_segments_embeddings - anchors).pow(2).sum(-1)
    )  # (batch/2, 32, embed_dim)
    tp_frame_inds = torch.argsort(d_to_anomalies, dim=1)[
        :, -top_anomaly_frames:
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
    # positive_frames = anomal_segments_embeddings[:, 1:1+n_anom_video, :].view(n_anom_video, -1, embed_dim)

    """
    Prepare negative (normal) frames embeddings
    To do that we pick N=top_normal_frames frames from normal videos that are distant from the anchors
    Here we calculate pairwise distance to pick hard negative examples for each anchor
    """
    n_norm_video, _, _ = normal_segments_embeddings.shape
    fp_frame_inds = torch.cdist(normal_segments_embeddings, anchors).argsort(dim=1)[
        :, -top_normal_frames:, 0
    ]
    fp_frame_mask = torch.zeros((n_norm_video, n_frames), dtype=torch.bool).to(
        anomal_segments_embeddings.device
    )
    for row_idx in range(fp_frame_inds.shape[0]):
        for col_idx in fp_frame_inds[row_idx]:
            fp_frame_mask[row_idx, col_idx] = True

    negative_frames = normal_segments_embeddings[fp_frame_mask].view(
        n_norm_video, -1, embed_dim
    )  # (batch/2,top_anomaly_frames, embed_dim)

    """
    Triplet loss
    """
    triplet_loss = (
        (positive_frames - anchors).pow(2).sum(-1).mean()
        - (negative_frames - anchors).pow(2).sum(-1).mean()
        + margin
    )
    triplet_loss = torch.relu(triplet_loss)
    """
    Smoothness of anomalous video
    """
    smoothed_scores = positive_frames[:, 1:] - positive_frames[:, :-1]
    smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    # sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (triplet_loss + lambdas * smoothed_scores_sum_squared).mean()
    return final_loss


def triplet_objective_sampling(
    normalized_embeddings,
    y_true,
    lambdas=8e-5,
    top_anomaly_frames=3,
    top_normal_frames=3,
    margin=0.2,
):
    # normalized_embeddings (batch_size, 32, embed_dim)
    # y_true (batch_size)

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    normal_segments_embeddings = normalized_embeddings[normal_vids_indices].squeeze(
        -1
    )  # (batch/2, 32, embed_dim)
    anomal_segments_embeddings = normalized_embeddings[anomal_vids_indices].squeeze(
        -1
    )  # (batch/2, 32, embed_dim)

    # anchors are the frame 0 of the normal video
    anchors = normal_segments_embeddings[:, :1, :]  # (batch/2, 1, embed_dim)

    """
    Prepare positive frames embeddings
    To do that we pick N frames from normal video that are near from the corresponding anchor
    """
    n_norm_video, n_frames, embed_dim = normal_segments_embeddings.shape
    positive_frames = normal_segments_embeddings[:, 1 : 1 + top_normal_frames, :].view(
        n_norm_video, -1, embed_dim
    )

    """
    Prepare negative (normal) frames embeddings
    To do that we pick N=top_normal_frames frames from normal videos that are distant from the anchors
    Here we calculate pairwise distance to pick hard negative examples for each anchor
    """
    n_anom_video, _, _ = anomal_segments_embeddings.shape
    fp_frame_inds = torch.cdist(anomal_segments_embeddings, anchors).argsort(dim=1)[
        :, -top_anomaly_frames:, 0
    ]
    fp_frame_mask = torch.zeros((n_anom_video, n_frames), dtype=torch.bool).to(
        anomal_segments_embeddings.device
    )
    for row_idx in range(fp_frame_inds.shape[0]):
        for col_idx in fp_frame_inds[row_idx]:
            fp_frame_mask[row_idx, col_idx] = True

    negative_frames = anomal_segments_embeddings[fp_frame_mask].view(
        n_anom_video, -1, embed_dim
    )  # (batch/2,top_anomaly_frames, embed_dim)

    """
    Triplet loss
    """
    triplet_loss = (
        (positive_frames - anchors).pow(2).sum(-1).mean()
        - (negative_frames - anchors).pow(2).sum(-1).mean()
        + margin
    )
    triplet_loss = torch.relu(triplet_loss)
    """
    Smoothness of anomalous video
    """
    smoothed_scores = (
        anomal_segments_embeddings[:, 1:] - anomal_segments_embeddings[:, :-1]
    )
    smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    # sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (triplet_loss + lambdas * smoothed_scores_sum_squared).mean()
    return final_loss
