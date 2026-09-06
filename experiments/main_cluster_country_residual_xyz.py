import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.dataloaders_cluster_country_residual_xyz import (
    create_cluster_country_residual_xyz_dataloaders,
)
from src.models.model_cluster_country_residual_xyz import (
    GeoClusterCountryResidualXYZModel,
)
from src.training.train_cluster_country_residual_xyz import (
    train_one_epoch_cluster_country_residual_xyz,
    validate_cluster_country_residual_xyz,
)
from src.utils.metrics import haversine_distance_km


RESIDUAL_SCALE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "K=128 geographic cluster + country + "
            "bounded spherical XYZ residual experiment."
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
        "--country-weight",
        type=float,
        default=0.2,
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
        "--pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional baseline cluster-country checkpoint "
            "used to initialize the backbone, cluster head, "
            "and country head."
        ),
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=(
            "cluster128_country_residual_xyz_"
            "scaled_rw1_seed42.pth"
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
        print(
            "Warning: expected only the "
            "residual head to remain new."
        )


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    k = args.num_clusters

    print(
        "Model: MobileNetV3-Large "
        "cluster + auxiliary country "
        "+ bounded spherical XYZ residual"
    )

    print(
        "Pretrained:",
        args.pretrained,
    )

    print(
        "Clusters:",
        k,
    )

    print(
        "Countries:",
        args.num_countries,
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
        "Country weight:",
        args.country_weight,
    )

    print(
        "Residual weight:",
        args.residual_weight,
    )

    print(
        "Residual scale:",
        RESIDUAL_SCALE,
    )

    print(
        "Label smoothing:",
        args.label_smoothing,
    )

    print(
        "Random seed:",
        args.seed,
    )

    print(
        "Initialization checkpoint:",
        args.init_checkpoint,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device,
    )

    train_csv = (
        f"geo_dataset/"
        f"train_cluster{k}.csv"
    )

    val_csv = (
        f"geo_dataset/"
        f"val_cluster{k}.csv"
    )

    centers_path = (
        f"geo_dataset/"
        f"cluster{k}_centers_xyz.npy"
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
            pretrained=args.pretrained,
        )
    )

    model = GeoClusterCountryResidualXYZModel(
        num_clusters=k,
        num_countries=args.num_countries,
        pretrained=args.pretrained,
    ).to(
        device
    )

    if args.init_checkpoint is not None:
        load_baseline_checkpoint(
            model=model,
            checkpoint_path=(
                args.init_checkpoint
            ),
            device=device,
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
        "Under 5 million:",
        num_params
        <= 5_000_000,
    )

    if num_params > 5_000_000:
        raise RuntimeError(
            "Model exceeds the "
            "5,000,000 parameter limit."
        )

    centers_xyz = np.load(
        centers_path
    ).astype(
        np.float32
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

    val_data = np.genfromtxt(
        val_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    true_coordinates = (
        np.column_stack(
            [
                val_data["lat"],
                val_data["lng"],
            ]
        )
    )

    true_coordinates = torch.tensor(
        true_coordinates,
        dtype=torch.float32,
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
        train_metrics = (
            train_one_epoch_cluster_country_residual_xyz(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                country_weight=(
                    args.country_weight
                ),
                residual_weight=(
                    args.residual_weight
                ),
                label_smoothing=(
                    args.label_smoothing
                ),
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

        # Convert the normalized residual
        # prediction back into XYZ units.
        predicted_residuals = (
            RESIDUAL_SCALE
            * predicted_residuals_scaled
        )

        predicted_centers_xyz = (
            centers_xyz[
                predicted_clusters
            ]
        )

        predicted_xyz = (
            predicted_centers_xyz
            + predicted_residuals
        )

        norms = np.linalg.norm(
            predicted_xyz,
            axis=1,
            keepdims=True,
        )

        predicted_xyz = (
            predicted_xyz
            / np.clip(
                norms,
                a_min=1e-12,
                a_max=None,
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

        if median_km < best_median_km:
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
    