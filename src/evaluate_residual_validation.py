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


RESIDUAL_SCALE = 0.05


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the K=128 cluster-country "
            "spherical residual model."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        default="geo_dataset/train_cluster128.csv",
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        default="geo_dataset/val_cluster128.csv",
    )

    parser.add_argument(
        "--centers",
        type=str,
        default="geo_dataset/cluster128_centers_xyz.npy",
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="geo_dataset/train",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)

    centers_xyz = np.load(
        args.centers
    ).astype(np.float32)

    _, val_loader = (
        create_cluster_country_residual_xyz_dataloaders(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            image_dir=args.image_dir,
            centers_path=args.centers,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pretrained=True,
        )
    )

    model = GeoClusterCountryResidualXYZModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
        pretrained=False,
    ).to(device)

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    val_metrics = (
        validate_cluster_country_residual_xyz(
            model=model,
            dataloader=val_loader,
            device=device,
        )
    )

    predicted_clusters = (
        val_metrics["cluster_predictions"]
        .numpy()
    )

    predicted_residuals_scaled = (
        val_metrics["residual_predictions"]
        .numpy()
    )

    predicted_residuals = (
        RESIDUAL_SCALE
        * predicted_residuals_scaled
    )

    predicted_xyz = (
        centers_xyz[predicted_clusters]
        + predicted_residuals
    )

    predicted_xyz = (
        predicted_xyz
        / np.clip(
            np.linalg.norm(
                predicted_xyz,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )
    )

    predicted_coordinates = (
        xyz_to_latlng(
            predicted_xyz
        )
    )

    val_data = np.genfromtxt(
        args.val_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    true_coordinates = np.column_stack(
        [
            val_data["lat"],
            val_data["lng"],
        ]
    )

    predicted_coordinates = torch.tensor(
        predicted_coordinates,
        dtype=torch.float32,
    )

    true_coordinates = torch.tensor(
        true_coordinates,
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    )

    median_km = torch.median(
        distances
    ).item()

    mean_km = torch.mean(
        distances
    ).item()

    within_200 = (
        (distances < 200)
        .float()
        .mean()
        .item()
        * 100
    )

    within_750 = (
        (distances < 750)
        .float()
        .mean()
        .item()
        * 100
    )

    print()
    print("=== Residual Validation Results ===")
    print(
        f"Median distance: {median_km:.2f} km"
    )
    print(
        f"Mean distance: {mean_km:.2f} km"
    )
    print(
        f"<200 km: {within_200:.2f}%"
    )
    print(
        f"<750 km: {within_750:.2f}%"
    )
    print(
        f"Cluster accuracy: "
        f"{val_metrics['cluster_accuracy'] * 100:.2f}%"
    )
    print(
        f"Country accuracy: "
        f"{val_metrics['country_accuracy'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
    