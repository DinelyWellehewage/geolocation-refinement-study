import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-baseline K=128 cluster-country model "
            "with trainable bounded spherical XYZ residual head."
        )
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
        default=10,
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
        "--init-checkpoint",
        type=str,
        default=(
            "checkpoints/cluster_country/"
            "validation_k128_cw020.pth"
        ),
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=(
            "k128_residual_xyz_frozen_baseline.pth"
        ),
    )

    return parser.parse_args()


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def xyz_to_latlng(
    xyz,
):
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
        x**2
        + y**2
    )

    lat = np.arctan2(
        z,
        hyp,
    )

    return np.stack(
        [
            np.degrees(
                lat
            ),
            np.degrees(
                lng
            ),
        ],
        axis=1,
    )


def load_baseline_checkpoint(
    model,
    checkpoint_path,
    device,
):
    print(
        "Initializing from baseline checkpoint:",
        checkpoint_path,
    )

    baseline_state = torch.load(
        checkpoint_path,
        map_location=device,
    )

    current_state = model.state_dict()

    compatible_state = {
        key: value
        for key, value in baseline_state.items()
        if (
            key in current_state
            and current_state[key].shape
            == value.shape
        )
    }

    missing_keys = sorted(
        set(
            current_state.keys()
        )
        - set(
            compatible_state.keys()
        )
    )

    unexpected_keys = sorted(
        set(
            baseline_state.keys()
        )
        - set(
            current_state.keys()
        )
    )

    model.load_state_dict(
        compatible_state,
        strict=False,
    )

    print(
        "Loaded baseline tensors:",
        len(
            compatible_state
        ),
    )

    print(
        "New/uninitialized tensors:",
        missing_keys,
    )

    if unexpected_keys:
        print(
            "Unused baseline tensors:",
            unexpected_keys,
        )

    expected_new = {
        "residual_head.weight",
        "residual_head.bias",
    }

    if set(missing_keys) != expected_new:
        raise RuntimeError(
            "Baseline initialization mismatch. "
            "Expected only residual head parameters "
            "to remain newly initialized."
        )


def freeze_baseline(
    model,
):
    for (
        name,
        parameter,
    ) in model.named_parameters():
        if name.startswith(
            "residual_head."
        ):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False

    trainable_names = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        trainable_names,
    )

    print(
        "Trainable parameter count:",
        trainable_count,
    )

    expected_trainable = {
        "residual_head.weight",
        "residual_head.bias",
    }

    if set(
        trainable_names
    ) != expected_trainable:
        raise RuntimeError(
            "Unexpected trainable parameters."
        )


def train_one_epoch_frozen_residual(
    model,
    dataloader,
    optimizer,
    device,
):
    model.train()

    residual_loss_fn = (
        nn.SmoothL1Loss()
    )

    total_residual_loss = 0.0

    for (
        images,
        _,
        _,
        residual_targets,
    ) in dataloader:
        images = images.to(
            device
        )

        residual_targets = (
            residual_targets.to(
                device
            )
        )

        optimizer.zero_grad()

        (
            _,
            _,
            residual_predictions,
        ) = model(
            images
        )

        residual_loss = (
            residual_loss_fn(
                residual_predictions,
                residual_targets,
            )
        )

        residual_loss.backward()

        optimizer.step()

        total_residual_loss += (
            residual_loss.item()
        )

    return (
        total_residual_loss
        / len(
            dataloader
        )
    )


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Frozen-baseline spherical residual experiment"
    )

    print(
        "Using device:",
        device,
    )

    print(
        "Clusters:",
        args.num_clusters,
    )

    print(
        "Countries:",
        args.num_countries,
    )

    print(
        "Residual scale:",
        RESIDUAL_SCALE,
    )

    print(
        "Learning rate:",
        args.lr,
    )

    print(
        "Weight decay:",
        args.weight_decay,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "Random seed:",
        args.seed,
    )

    train_csv = (
        "geo_dataset/"
        f"train_cluster{args.num_clusters}.csv"
    )

    val_csv = (
        "geo_dataset/"
        f"val_cluster{args.num_clusters}.csv"
    )

    centers_path = (
        "geo_dataset/"
        f"cluster{args.num_clusters}_centers_xyz.npy"
    )

    (
        train_loader,
        val_loader,
    ) = (
        create_cluster_country_residual_xyz_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            image_dir="geo_dataset/train",
            centers_path=centers_path,
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
    ).to(
        device
    )

    load_baseline_checkpoint(
        model=model,
        checkpoint_path=(
            args.init_checkpoint
        ),
        device=device,
    )

    freeze_baseline(
        model
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Total parameter count:",
        total_params,
    )

    print(
        "Trainable parameter count:",
        trainable_params,
    )

    print(
        "Under 5 million:",
        total_params
        <= 5_000_000,
    )

    centers_xyz = np.load(
        centers_path
    ).astype(
        np.float32
    )

    val_data = np.genfromtxt(
        val_csv,
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

    true_coordinates = torch.tensor(
        true_coordinates,
        dtype=torch.float32,
    )

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = (
        "checkpoints/"
        "cluster_country_residual_xyz"
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
        residual_loss = (
            train_one_epoch_frozen_residual(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
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

        predicted_residuals_scaled = (
            val_metrics[
                "residual_predictions"
            ]
            .numpy()
        )

        predicted_residuals = (
            RESIDUAL_SCALE
            * predicted_residuals_scaled
        )

        predicted_xyz = (
            centers_xyz[
                predicted_clusters
            ]
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

        predicted_coordinates = (
            torch.tensor(
                predicted_coordinates,
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

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Residual Loss: "
            f"{residual_loss:.6f} "
            f"| Val Cluster Acc: "
            f"{val_metrics['cluster_accuracy'] * 100:.2f}% "
            f"| Val Country Acc: "
            f"{val_metrics['country_accuracy'] * 100:.2f}% "
            f"| Median: "
            f"{median_km:.2f} km "
            f"| Mean: "
            f"{mean_km:.2f} km "
            f"| <200: "
            f"{within_200:.2f}% "
            f"| <750: "
            f"{within_750:.2f}%"
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
                "Saved new best frozen-residual model "
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
    