import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.dataloaders_cluster_country import (
    create_cluster_country_dataloaders,
)
from src.models.model_cluster_country import (
    GeoClusterCountryModel,
)
from src.training.train_cluster_country import (
    train_one_epoch_cluster_country,
    validate_cluster_country,
)
from src.utils.metrics import haversine_distance_km


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
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
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--country-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
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
        "--experiment-name",
        type=str,
        default="cv_k128",
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
    xyz = (
        xyz
        / np.linalg.norm(
            xyz,
            axis=1,
            keepdims=True,
        )
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
    set_seed(args.seed)

    fold = args.fold
    k = args.num_clusters

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Fold:", fold)
    print("Clusters:", k)
    print("Using device:", device)
    print("Pretrained: True")
    print(
        "Country weight:",
        args.country_weight,
    )
    print(
        "Weight decay:",
        args.weight_decay,
    )
    print(
        "Label smoothing:",
        args.label_smoothing,
    )
    print(
        "Experiment:",
        args.experiment_name,
    )

    base_dir = (
        f"geo_dataset/cv_cluster{k}"
    )

    train_csv = os.path.join(
        base_dir,
        f"train_fold{fold}.csv",
    )

    val_csv = os.path.join(
        base_dir,
        f"val_fold{fold}.csv",
    )

    centers_path = os.path.join(
        base_dir,
        f"centers_fold{fold}.npy",
    )

    train_loader, val_loader = (
        create_cluster_country_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            image_dir="geo_dataset/train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pretrained=True,
        )
    )

    model = GeoClusterCountryModel(
        num_clusters=k,
        num_countries=args.num_countries,
        pretrained=True,
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameter count:",
        num_params,
    )

    print(
        "Under 5 million:",
        num_params <= 5_000_000,
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

    centers_xyz = np.load(
        centers_path
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
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

    checkpoint_dir = os.path.join(
        "checkpoints",
        args.experiment_name,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"fold{fold}.pth",
    )

    result_dir = os.path.join(
        "results",
        args.experiment_name,
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    prediction_path = os.path.join(
        result_dir,
        f"predictions_fold{fold}.csv",
    )

    best_median = float(
        "inf"
    )

    best_predictions = None

    for epoch in range(
        args.epochs
    ):
        train_metrics = (
            train_one_epoch_cluster_country(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                country_weight=(
                    args.country_weight
                ),
                consistency_weight=0.0,
                consistency_loss_fn=None,
                smoothing_mode="ordinary",
                centers_xyz=centers_xyz,
                label_smoothing=(
                    args.label_smoothing
                ),
                geographic_temperature_km=500.0,
            )
        )

        val_metrics = (
            validate_cluster_country(
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

        predicted_latlng = (
            centers_latlng[
                predicted_clusters
            ]
        )

        predicted_coordinates = (
            torch.tensor(
                predicted_latlng,
                dtype=torch.float32,
            )
        )

        distances = (
            haversine_distance_km(
                predicted_coordinates,
                true_coordinates,
            )
        )

        median_km = (
            torch.median(
                distances
            ).item()
        )

        mean_km = (
            torch.mean(
                distances
            ).item()
        )

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

        scheduler.step(
            median_km
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Train Cluster Acc: "
            f"{train_metrics['cluster_accuracy'] * 100:.2f}% "
            f"| Train Country Acc: "
            f"{train_metrics['country_accuracy'] * 100:.2f}% "
            f"| Val Cluster Acc: "
            f"{val_metrics['cluster_accuracy'] * 100:.2f}% "
            f"| Val Country Acc: "
            f"{val_metrics['country_accuracy'] * 100:.2f}% "
            f"| Median: {median_km:.2f} km "
            f"| Mean: {mean_km:.2f} km "
            f"| <200: {within_200:.2f}% "
            f"| <750: {within_750:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        if (
            median_km
            < best_median
        ):
            best_median = (
                median_km
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            best_predictions = (
                predicted_latlng.copy()
            )

            print(
                f"Saved new best fold {fold} model "
                f"({best_median:.2f} km) "
                f"to {checkpoint_path}"
            )

    if best_predictions is None:
        raise RuntimeError(
            "No best predictions were saved."
        )

    output_df = val_df[
        [
            "filename",
            "country",
            "lat",
            "lng",
            "fold",
        ]
    ].copy()

    output_df[
        "pred_lat"
    ] = best_predictions[
        :,
        0,
    ]

    output_df[
        "pred_lng"
    ] = best_predictions[
        :,
        1,
    ]

    pred_tensor = torch.tensor(
        best_predictions,
        dtype=torch.float32,
    )

    final_distances = (
        haversine_distance_km(
            pred_tensor,
            true_coordinates,
        )
        .numpy()
    )

    output_df[
        "distance_km"
    ] = final_distances

    output_df.to_csv(
        prediction_path,
        index=False,
    )

    final_median = float(
        np.median(
            final_distances
        )
    )

    final_mean = float(
        np.mean(
            final_distances
        )
    )

    final_within_200 = float(
        (
            final_distances < 200
        ).mean()
        * 100
    )

    final_within_750 = float(
        (
            final_distances < 750
        ).mean()
        * 100
    )

    print()
    print(
        f"=== Best fold {fold} metrics ==="
    )

    print(
        "Median:",
        f"{final_median:.2f} km",
    )

    print(
        "Mean:",
        f"{final_mean:.2f} km",
    )

    print(
        "<200:",
        f"{final_within_200:.2f}%",
    )

    print(
        "<750:",
        f"{final_within_750:.2f}%",
    )

    print()
    print(
        f"Best fold {fold} median:",
        f"{best_median:.2f} km",
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Saved predictions:",
        prediction_path,
    )


if __name__ == "__main__":
    main()