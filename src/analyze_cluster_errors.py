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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/cluster64/"
            "cluster64_strongaug_seed42.pth"
        ),
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
    )

    model = GeoClusterModel(
        num_clusters=args.num_clusters,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
        )
    )

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, clusters in val_loader:
            images = images.to(device)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                clusters.cpu()
            )

    predicted_clusters = torch.cat(
        all_predictions
    ).numpy()

    true_clusters = torch.cat(
        all_targets
    ).numpy()

    centers_xyz = np.load(
        "geo_dataset/cluster64_centers_xyz.npy"
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    val_df = pd.read_csv(
        "geo_dataset/val_cluster64.csv"
    )

    true_coordinates = torch.tensor(
        val_df[
            ["lat", "lng"]
        ].values,
        dtype=torch.float32,
    )

    predicted_coordinates = torch.tensor(
        centers_latlng[
            predicted_clusters
        ],
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    ).numpy()

    correct_cluster = (
        predicted_clusters
        == true_clusters
    )

    wrong_cluster = ~correct_cluster

    near_wrong = (
        wrong_cluster
        & (distances < 200)
    )

    medium_wrong = (
        wrong_cluster
        & (distances >= 200)
        & (distances < 750)
    )

    catastrophic_wrong = (
        wrong_cluster
        & (distances >= 750)
    )

    total = len(distances)

    def pct(mask):
        return (
            mask.sum()
            / total
            * 100
        )

    print()
    print("=== Error breakdown ===")

    print(
        f"Correct cluster: "
        f"{correct_cluster.sum()} "
        f"({pct(correct_cluster):.2f}%)"
    )

    print(
        f"Wrong cluster but <200 km: "
        f"{near_wrong.sum()} "
        f"({pct(near_wrong):.2f}%)"
    )

    print(
        f"Wrong cluster 200-750 km: "
        f"{medium_wrong.sum()} "
        f"({pct(medium_wrong):.2f}%)"
    )

    print(
        f"Wrong cluster >=750 km: "
        f"{catastrophic_wrong.sum()} "
        f"({pct(catastrophic_wrong):.2f}%)"
    )

    print()
    print("=== Distance metrics ===")

    print(
        "Median:",
        f"{np.median(distances):.2f} km",
    )

    print(
        "Mean:",
        f"{np.mean(distances):.2f} km",
    )

    print(
        "<200 km:",
        f"{(distances < 200).mean() * 100:.2f}%",
    )

    print(
        "<750 km:",
        f"{(distances < 750).mean() * 100:.2f}%",
    )

    confusion = np.zeros(
        (
            args.num_clusters,
            args.num_clusters,
        ),
        dtype=np.int64,
    )

    for true_cluster, pred_cluster in zip(
        true_clusters,
        predicted_clusters,
    ):
        confusion[
            true_cluster,
            pred_cluster,
        ] += 1

    np.save(
        "geo_dataset/"
        "cluster64_confusion.npy",
        confusion,
    )

    results = val_df.copy()

    results[
        "predicted_cluster"
    ] = predicted_clusters

    results[
        "distance_km"
    ] = distances

    results[
        "correct_cluster"
    ] = correct_cluster

    results[
        "error_category"
    ] = np.select(
        [
            correct_cluster,
            near_wrong,
            medium_wrong,
            catastrophic_wrong,
        ],
        [
            "correct_cluster",
            "wrong_under_200",
            "wrong_200_750",
            "wrong_over_750",
        ],
        default="unknown",
    )

    results.to_csv(
        "geo_dataset/"
        "cluster64_error_analysis.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(
        "geo_dataset/cluster64_confusion.npy"
    )
    print(
        "geo_dataset/"
        "cluster64_error_analysis.csv"
    )


if __name__ == "__main__":
    main()