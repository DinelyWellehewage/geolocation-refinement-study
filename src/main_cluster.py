import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.dataloaders_cluster import create_cluster_dataloaders
from src.models.model_cluster import GeoClusterModel
from src.training.train_cluster import (
    train_one_epoch_cluster,
    validate_cluster,
)
from src.utils.metrics import haversine_distance_km


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="cluster64_seed42.pth",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    args = parse_args()
    set_seed(args.seed)

    print("Model: MobileNetV3-Large 64-cluster classifier")
    print("Random seed:", args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader = create_cluster_dataloaders(
        train_csv="geo_dataset/train_cluster64.csv",
        val_csv="geo_dataset/val_cluster64.csv",
        image_dir="geo_dataset/train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = GeoClusterModel(
        num_clusters=64,
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameter count:", num_params)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    centers_xyz = np.load(
        "geo_dataset/cluster64_centers_xyz.npy"
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    val_csv = np.genfromtxt(
        "geo_dataset/val_cluster64.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    true_coordinates = np.column_stack(
        [
            val_csv["lat"],
            val_csv["lng"],
        ]
    )

    true_coordinates = torch.tensor(
        true_coordinates,
        dtype=torch.float32,
    )

    os.makedirs(
        "checkpoints/cluster",
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        "checkpoints/cluster",
        args.checkpoint_name,
    )

    best_median_km = float("inf")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch_cluster(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_cluster(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        predicted_clusters = (
            val_metrics["predictions"].numpy()
        )

        predicted_coordinates = centers_latlng[
            predicted_clusters
        ]

        predicted_coordinates = torch.tensor(
            predicted_coordinates,
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

        scheduler.step(median_km)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Train Loss: {train_metrics['loss']:.4f} "
            f"| Train Acc: {train_metrics['accuracy'] * 100:.2f}% "
            f"| Val Cluster Acc: {val_metrics['accuracy'] * 100:.2f}% "
            f"| Median: {median_km:.2f} km "
            f"| Mean: {mean_km:.2f} km "
            f"| <200: {within_200:.2f}% "
            f"| <750: {within_750:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        if median_km < best_median_km:
            best_median_km = median_km

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Saved new best cluster model "
                f"({best_median_km:.2f} km) "
                f"to {checkpoint_path}"
            )


if __name__ == "__main__":
    main()