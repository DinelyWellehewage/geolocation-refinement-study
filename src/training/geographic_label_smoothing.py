import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def xyz_to_latlng(xyz):
    xyz = xyz / np.linalg.norm(
        xyz,
        axis=1,
        keepdims=True,
    )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lng = np.arctan2(y, x)
    hyp = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, hyp)

    return np.stack(
        [
            np.degrees(lat),
            np.degrees(lng),
        ],
        axis=1,
    )


def pairwise_haversine_km(latlng):
    lat = np.radians(
        latlng[:, 0]
    )

    lng = np.radians(
        latlng[:, 1]
    )

    lat1 = lat[:, None]
    lat2 = lat[None, :]

    lng1 = lng[:, None]
    lng2 = lng[None, :]

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlng / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(
        np.sqrt(
            np.clip(
                a,
                0.0,
                1.0,
            )
        )
    )

    earth_radius_km = 6371.0088

    return (
        earth_radius_km
        * c
    )


def build_geographic_targets(
    centers_xyz,
    smoothing=0.1,
    temperature_km=500.0,
):
    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    distances = pairwise_haversine_km(
        centers_latlng
    )

    num_clusters = len(
        centers_xyz
    )

    target_matrix = np.zeros(
        (
            num_clusters,
            num_clusters,
        ),
        dtype=np.float32,
    )

    for true_cluster in range(
        num_clusters
    ):
        row_distances = distances[
            true_cluster
        ].copy()

        weights = np.exp(
            -row_distances
            / temperature_km
        )

        # Do not allocate smoothing probability
        # to the true class here.
        weights[
            true_cluster
        ] = 0.0

        weight_sum = weights.sum()

        if weight_sum > 0:
            weights = (
                weights
                / weight_sum
            )

        target_matrix[
            true_cluster
        ] = (
            smoothing
            * weights
        )

        target_matrix[
            true_cluster,
            true_cluster,
        ] = (
            1.0 - smoothing
        )

    return target_matrix


class GeographicLabelSmoothingLoss(
    nn.Module
):
    def __init__(
        self,
        centers_xyz,
        smoothing=0.1,
        temperature_km=500.0,
    ):
        super().__init__()

        target_matrix = (
            build_geographic_targets(
                centers_xyz=centers_xyz,
                smoothing=smoothing,
                temperature_km=temperature_km,
            )
        )

        self.register_buffer(
            "target_matrix",
            torch.tensor(
                target_matrix,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits,
        targets,
    ):
        soft_targets = (
            self.target_matrix[
                targets
            ]
        )

        log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        loss = -(
            soft_targets
            * log_probs
        ).sum(
            dim=1
        )

        return loss.mean()