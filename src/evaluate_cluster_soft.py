import argparse

import numpy as np
import pandas as pd
import torch

from src.datasets.dataloaders_cluster import create_cluster_dataloaders
from src.models.model_cluster import GeoClusterModel
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


def weighted_xyz_prediction(
    probabilities,
    centers_xyz,
    top_k,
):
    top_values, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1,
    )

    top_values = top_values / top_values.sum(
        dim=1,
        keepdim=True,
    )

    centers = torch.tensor(
        centers_xyz,
        dtype=torch.float32,
    )

    selected_centers = centers[
        top_indices.cpu()
    ]

    weighted_xyz = (
        selected_centers
        * top_values.cpu().unsqueeze(-1)
    ).sum(dim=1)

    weighted_xyz = weighted_xyz / torch.linalg.norm(
        weighted_xyz,
        dim=1,
        keepdim=True,
    )

    return weighted_xyz.numpy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/cluster/cluster64_seed42.pth",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)

    _, val_loader = create_cluster_dataloaders(
        train_csv="geo_dataset/train_cluster64.csv",
        val_csv="geo_dataset/val_cluster64.csv",
        image_dir="geo_dataset/train",
        batch_size=64,
        num_workers=0,
        seed=42,
    )

    model = GeoClusterModel(
        num_clusters=64,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
        )
    )

    model.eval()

    all_probabilities = []

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            all_probabilities.append(
                probabilities.cpu()
            )

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )

    centers_xyz = np.load(
        "geo_dataset/cluster64_centers_xyz.npy"
    )

    val_df = pd.read_csv(
        "geo_dataset/val_cluster64.csv"
    )

    targets = torch.tensor(
        val_df[
            ["lat", "lng"]
        ].values,
        dtype=torch.float32,
    )

    print()
    print(
        f"{'top_k':>6} "
        f"{'median_km':>12} "
        f"{'mean_km':>12} "
        f"{'<200km':>10} "
        f"{'<750km':>10}"
    )

    for top_k in [1, 2, 3, 5, 8, 10]:
        predicted_xyz = weighted_xyz_prediction(
            probabilities,
            centers_xyz,
            top_k,
        )

        predicted_latlng = xyz_to_latlng(
            predicted_xyz
        )

        predictions = torch.tensor(
            predicted_latlng,
            dtype=torch.float32,
        )

        distances = haversine_distance_km(
            predictions,
            targets,
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

        print(
            f"{top_k:6d} "
            f"{median_km:12.2f} "
            f"{mean_km:12.2f} "
            f"{within_200:9.2f}% "
            f"{within_750:9.2f}%"
        )


if __name__ == "__main__":
    main()