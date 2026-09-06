import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW

from src.datasets.dataloaders_cluster_country_residual_xyz import (
    create_cluster_country_residual_xyz_dataloaders,
)
from src.models.model_cluster_country_residual_xyz import (
    GeoClusterCountryResidualXYZModel,
)
from src.training.train_cluster_country_residual_xyz import (
    train_one_epoch_cluster_country_residual_xyz,
)


RESIDUAL_SCALE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Final full-data K=128 cluster + country + "
            "bounded spherical XYZ residual training."
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
        default=6,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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
            "checkpoints/final/"
            "final_k128_imagenet_cw020.pth"
        ),
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=(
            "final_k128_residual_xyz_"
            "baseline_init_rw1.pth"
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


def load_baseline_checkpoint(
    model,
    checkpoint_path,
    device,
):
    print(
        "Initializing from full-data baseline checkpoint:",
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
        set(current_state.keys())
        - set(compatible_state.keys())
    )

    unexpected_keys = sorted(
        set(baseline_state.keys())
        - set(current_state.keys())
    )

    model.load_state_dict(
        compatible_state,
        strict=False,
    )

    print(
        "Loaded baseline tensors:",
        len(compatible_state),
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
            "Expected only residual_head.weight "
            "and residual_head.bias to be new."
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
        "Final full-data bounded spherical "
        "residual fine-tuning"
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
        "Pretrained transforms: True"
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
        "Weight decay:",
        args.weight_decay,
    )

    print(
        "Label smoothing:",
        args.label_smoothing,
    )

    print(
        "Learning rate:",
        args.lr,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "Initialization checkpoint:",
        args.init_checkpoint,
    )

    train_csv = (
        "geo_dataset/final/"
        f"train_cluster{args.num_clusters}_full.csv"
    )

    centers_path = (
        "geo_dataset/final/"
        f"cluster{args.num_clusters}_centers_full.npy"
    )

    # Full-data training has no validation split.
    # Reuse the training CSV for the unused validation
    # loader, matching the existing final baseline runner.
    train_loader, _ = (
        create_cluster_country_residual_xyz_dataloaders(
            train_csv=train_csv,
            val_csv=train_csv,
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

        # The complete backbone will immediately be
        # overwritten by the saved full-data baseline.
        # Avoid an unnecessary ImageNet weight load.
        pretrained=False,
    ).to(device)

    load_baseline_checkpoint(
        model=model,
        checkpoint_path=args.init_checkpoint,
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
        num_params <= 5_000_000,
    )

    if num_params > 5_000_000:
        raise RuntimeError(
            "Model exceeds the "
            "5,000,000 parameter limit."
        )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = (
        "checkpoints/final"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        args.checkpoint_name,
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
                country_weight=args.country_weight,
                residual_weight=args.residual_weight,
                label_smoothing=args.label_smoothing,
            )
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
            f"{train_metrics['country_accuracy'] * 100:.2f}%"
        )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print()
    print(
        "Final residual model saved:"
    )
    print(
        checkpoint_path
    )


if __name__ == "__main__":
    main()
    