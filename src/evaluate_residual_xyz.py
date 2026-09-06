import argparse

import numpy as np
import torch

from src.datasets.dataloaders_cluster_country_residual_xyz import (
    create_cluster_country_residual_xyz_dataloaders,
)
from src.models.model_cluster_country_residual_xyz import (
    GeoClusterCountryResidualXYZModel,
)
from src.training.train_cluster_country_residual_xyz import (
    validate_cluster_country_residual_xyz,
)
from src.utils.metrics import haversine_distance_km


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


def compute_metrics(
    predicted_xyz,
    true_coordinates,
):
    predicted_coordinates = xyz_to_latlng(
        predicted_xyz
    )

    predicted_coordinates = torch.tensor(
        predicted_coordinates,
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    )

    return {
        "median": torch.median(
            distances
        ).item(),
        "mean": torch.mean(
            distances
        ).item(),
        "within_200": (
            (distances < 200)
            .float()
            .mean()
            .item()
            * 100
        ),
        "within_750": (
            (distances < 750)
            .float()
            .mean()
            .item()
            * 100
        ),
    }


def print_metrics(
    name,
    metrics,
):
    print()
    print(name)
    print(
        f"Median: {metrics['median']:.2f} km"
    )
    print(
        f"Mean: {metrics['mean']:.2f} km"
    )
    print(
        f"<200 km: {metrics['within_200']:.2f}%"
    )
    print(
        f"<750 km: {metrics['within_750']:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-countries",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    centers_path = (
        "geo_dataset/"
        "cluster128_centers_xyz.npy"
    )

    train_csv = (
        "geo_dataset/"
        "train_cluster128.csv"
    )

    val_csv = (
        "geo_dataset/"
        "val_cluster128.csv"
    )

    centers_xyz = np.load(
        centers_path
    ).astype(np.float32)

    _, val_loader = (
        create_cluster_country_residual_xyz_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            image_dir="geo_dataset/train",
            centers_path=centers_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=42,
            pretrained=True,
        )
    )

    model = GeoClusterCountryResidualXYZModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
        )
    )

    val_metrics = (
        validate_cluster_country_residual_xyz(
            model=model,
            dataloader=val_loader,
            device=device,
        )
    )

    predicted_clusters = (
        val_metrics[
            "cluster_predictions"
        ]
        .numpy()
    )

    residual_predictions = (
        val_metrics[
            "residual_predictions"
        ]
        .numpy()
    )

    cluster_only_xyz = (
        centers_xyz[
            predicted_clusters
        ]
    )

    residual_xyz = (
        cluster_only_xyz
        + residual_predictions
    )

    residual_xyz = (
        residual_xyz
        / np.linalg.norm(
            residual_xyz,
            axis=1,
            keepdims=True,
        )
    )

    val_data = np.genfromtxt(
        val_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    true_coordinates = torch.tensor(
        np.column_stack(
            [
                val_data["lat"],
                val_data["lng"],
            ]
        ),
        dtype=torch.float32,
    )

    cluster_metrics = compute_metrics(
        cluster_only_xyz,
        true_coordinates,
    )

    residual_metrics = compute_metrics(
        residual_xyz,
        true_coordinates,
    )

    print(
        "Cluster accuracy:",
        f"{val_metrics['cluster_accuracy'] * 100:.2f}%",
    )

    print(
        "Country accuracy:",
        f"{val_metrics['country_accuracy'] * 100:.2f}%",
    )

    print_metrics(
        "=== Cluster center only ===",
        cluster_metrics,
    )

    print_metrics(
        "=== Cluster center + residual ===",
        residual_metrics,
    )


if __name__ == "__main__":
    main()

