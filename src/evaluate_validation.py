import argparse

import numpy as np
import torch

from src.datasets.dataloaders_cluster_country import (
    create_cluster_country_dataloaders,
)
from src.models.model_cluster_country import (
    GeoClusterCountryModel,
)
from src.training.train_cluster_country import (
    validate_cluster_country,
)
from src.utils.metrics import (
    haversine_distance_km,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the cluster-country model on the held-out validation set."
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

    return parser.parse_args()


def xyz_to_latlng(xyz):
    xyz = xyz / np.linalg.norm(
        xyz,
        axis=1,
        keepdims=True,
    )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lng = np.arctan2(
        y,
        x,
    )

    hyp = np.sqrt(
        x**2 + y**2
    )

    lat = np.arctan2(
        z,
        hyp,
    )

    return np.stack(
        [
            np.degrees(lat),
            np.degrees(lng),
        ],
        axis=1,
    )


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Clusters:", args.num_clusters)
    print("Countries:", args.num_countries)

    _, val_loader = create_cluster_country_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=True,
    )

    model = GeoClusterCountryModel(
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

    centers_xyz = np.load(
        args.centers
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    val_metrics = validate_cluster_country(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    predicted_clusters = (
        val_metrics["cluster_predictions"]
        .numpy()
    )

    predicted_coordinates = (
        centers_latlng[
            predicted_clusters
        ]
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
    print("=== Validation Results ===")
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