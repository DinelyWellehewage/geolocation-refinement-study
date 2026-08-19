import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.dataloaders_multitask import (
    create_multitask_dataloaders,
)
from src.models.model_conditioned_mobilenet_large import (
    CountryConditionedMobileNetLarge,
)
from src.training.train_multitask import (
    train_one_epoch_multitask,
    validate_multitask,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--haversine-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--country-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="conditioned_mobilenet_large_seed42.pth"
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    set_seed(args.seed)

    print("Model: CountryConditionedMobileNetV3-Large")
    print("Random seed:", args.seed)
    print("Haversine weight:", args.haversine_weight)
    print("Country weight:", args.country_weight)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader = create_multitask_dataloaders(
        train_csv="geo_dataset/train_split.csv",
        val_csv="geo_dataset/val_split.csv",
        image_dir="geo_dataset/train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = CountryConditionedMobileNetLarge(
        num_countries=12,
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameter count:", num_params)
    print("Under 5 million:", num_params <= 5_000_000)

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

    os.makedirs(
        "checkpoints/conditioned",
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        "checkpoints/conditioned",
        args.checkpoint_name,
    )

    best_val_distance = float("inf")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch_multitask(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            haversine_weight=args.haversine_weight,
            country_weight=args.country_weight,
        )

        val_metrics = validate_multitask(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        val_median_km = val_metrics[
            "median_haversine_km"
        ]

        country_accuracy = val_metrics[
            "country_accuracy"
        ]

        scheduler.step(
            val_median_km
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Total Loss: {train_metrics['total_loss']:.4f} "
            f"| Coordinate Loss: {train_metrics['coordinate_loss']:.4f} "
            f"| Country Loss: {train_metrics['country_loss']:.4f} "
            f"| Val Median Haversine: {val_median_km:.2f} km "
            f"| Country Accuracy: {country_accuracy * 100:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        if val_median_km < best_val_distance:
            best_val_distance = val_median_km

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Saved new best conditioned EfficientNet model "
                f"({best_val_distance:.2f} km) "
                f"to {checkpoint_path}"
            )


if __name__ == "__main__":
    main()
