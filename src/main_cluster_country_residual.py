import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.dataloaders_cluster_country_residual import (
    create_cluster_country_residual_dataloaders,
)
from src.models.model_cluster_country_residual import (
    GeoClusterCountryResidualModel,
)
from src.training.country_consistency import (
    CountryConsistencyLoss,
    build_cluster_country_matrix,
)
from src.training.train_cluster_country_residual import (
    train_one_epoch_cluster_country_residual,
    validate_cluster_country_residual,
)
from src.utils.metrics import haversine_distance_km


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-countries",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--country-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--residual-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.06,
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
        "--checkpoint-name",
        type=str,
        default=(
            "cluster64_country_residual_rw1_seed42.pth"
        ),
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


def evaluate_distances(
    predicted_xyz,
    true_coordinates,
):
    predicted_xyz = np.asarray(
        predicted_xyz,
        dtype=np.float32,
    )

    predicted_latlng = xyz_to_latlng(
        predicted_xyz
    )

    predicted_coordinates = torch.tensor(
        predicted_latlng,
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

    return {
        "median_km": median_km,
        "mean_km": mean_km,
        "within_200": within_200,
        "within_750": within_750,
        "distances": distances,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    k = args.num_clusters

    print(
        "Model: MobileNetV3-Large "
        "+ cluster head "
        "+ country head "
        "+ XYZ residual head"
    )

    print("Clusters:", k)
    print("Countries:", args.num_countries)
    print("Learning rate:", args.lr)
    print("Weight decay:", args.weight_decay)
    print("Country weight:", args.country_weight)
    print(
        "Consistency weight:",
        args.consistency_weight,
    )
    print(
        "Residual weight:",
        args.residual_weight,
    )
    print(
        "Residual scale:",
        args.residual_scale,
    )
    print(
        "Label smoothing:",
        args.label_smoothing,
    )
    print("Seed:", args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_csv = (
        f"geo_dataset/"
        f"train_cluster{k}_residual.csv"
    )

    val_csv = (
        f"geo_dataset/"
        f"val_cluster{k}_residual.csv"
    )

    centers_path = (
        f"geo_dataset/"
        f"cluster{k}_centers_xyz.npy"
    )

    train_loader, val_loader = (
        create_cluster_country_residual_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            image_dir="geo_dataset/train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    )

    centers_xyz = np.load(
        centers_path
    )

    model = (
        GeoClusterCountryResidualModel(
            num_clusters=k,
            num_countries=args.num_countries,
            residual_scale=args.residual_scale,
        )
        .to(device)
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameter count:",
        num_params,
    )

    print(
        "Under 5M:",
        num_params <= 5_000_000,
    )

    cluster_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=(
            args.label_smoothing
        ),
    ).to(device)

    cluster_country_matrix = (
        build_cluster_country_matrix(
            train_csv=train_csv,
            num_clusters=k,
            num_countries=args.num_countries,
        )
    )

    consistency_loss_fn = (
        CountryConsistencyLoss(
            cluster_country_matrix=(
                cluster_country_matrix
            ),
        )
        .to(device)
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    val_df = pd.read_csv(
        val_csv
    )

    true_coordinates = torch.tensor(
        val_df[
            [
                "lat",
                "lng",
            ]
        ].values,
        dtype=torch.float32,
    )

    checkpoint_dir = (
        "checkpoints/"
        "cluster_country_residual"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        args.checkpoint_name,
    )

    best_median_km = float(
        "inf"
    )

    for epoch in range(
        args.epochs
    ):
        train_metrics = (
            train_one_epoch_cluster_country_residual(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                cluster_loss_fn=cluster_loss_fn,
                consistency_loss_fn=(
                    consistency_loss_fn
                ),
                country_weight=(
                    args.country_weight
                ),
                consistency_weight=(
                    args.consistency_weight
                ),
                residual_weight=(
                    args.residual_weight
                ),
            )
        )

        val_metrics = (
            validate_cluster_country_residual(
                model=model,
                dataloader=val_loader,
                device=device,
                centers_xyz=centers_xyz,
            )
        )

        predicted_xyz = (
            val_metrics[
                "predicted_xyz"
            ]
            .numpy()
        )

        geo_metrics = evaluate_distances(
            predicted_xyz=predicted_xyz,
            true_coordinates=(
                true_coordinates
            ),
        )

        median_km = (
            geo_metrics[
                "median_km"
            ]
        )

        mean_km = (
            geo_metrics[
                "mean_km"
            ]
        )

        within_200 = (
            geo_metrics[
                "within_200"
            ]
        )

        within_750 = (
            geo_metrics[
                "within_750"
            ]
        )

        residual_predictions = (
            val_metrics[
                "residual_predictions"
            ]
        )

        residual_targets = (
            val_metrics[
                "residual_targets"
            ]
        )

        residual_mae = (
            torch.abs(
                residual_predictions
                - residual_targets
            )
            .mean()
            .item()
        )

        residual_rmse = (
            torch.sqrt(
                torch.mean(
                    (
                        residual_predictions
                        - residual_targets
                    )
                    ** 2
                )
            )
            .item()
        )

        scheduler.step(
            median_km
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Total Loss: "
            f"{train_metrics['total_loss']:.4f} "
            f"| Cluster Loss: "
            f"{train_metrics['cluster_loss']:.4f} "
            f"| Country Loss: "
            f"{train_metrics['country_loss']:.4f} "
            f"| Consistency Loss: "
            f"{train_metrics['consistency_loss']:.4f} "
            f"| Residual Loss: "
            f"{train_metrics['residual_loss']:.6f} "
            f"| Train Cluster Acc: "
            f"{train_metrics['cluster_accuracy'] * 100:.2f}% "
            f"| Train Country Acc: "
            f"{train_metrics['country_accuracy'] * 100:.2f}% "
            f"| Val Cluster Acc: "
            f"{val_metrics['cluster_accuracy'] * 100:.2f}% "
            f"| Val Country Acc: "
            f"{val_metrics['country_accuracy'] * 100:.2f}% "
            f"| Residual MAE: "
            f"{residual_mae:.6f} "
            f"| Residual RMSE: "
            f"{residual_rmse:.6f} "
            f"| Median: "
            f"{median_km:.2f} km "
            f"| Mean: "
            f"{mean_km:.2f} km "
            f"| <200: "
            f"{within_200:.2f}% "
            f"| <750: "
            f"{within_750:.2f}% "
            f"| LR: "
            f"{current_lr:.6f}"
        )

        if (
            median_km
            < best_median_km
        ):
            best_median_km = (
                median_km
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                "Saved new best "
                "cluster-country-residual model "
                f"({best_median_km:.2f} km) "
                f"to {checkpoint_path}"
            )

    print()
    print(
        "Best validation median:",
        f"{best_median_km:.2f} km",
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()