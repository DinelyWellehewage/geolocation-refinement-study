import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.dataloaders import create_dataloaders
from src.datasets.folds import create_fold_split
from src.models.model import GeoLocationModel
from src.training.train import train_one_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3, 4],
        required=True,
    )

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
        "--checkpoint-name",
        type=str,
        default=None,
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

    print("CV fold:", args.fold)
    print("Random seed:", args.seed)
    print("Haversine weight:", args.haversine_weight)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # ---------------------------------------------------------
    # Create fold-specific train / validation CSV files
    # ---------------------------------------------------------

    train_csv = f"geo_dataset/cv_train_fold{args.fold}.csv"
    val_csv = f"geo_dataset/cv_val_fold{args.fold}.csv"

    create_fold_split(
        folds_csv="folds.csv",
        fold=args.fold,
        train_output=train_csv,
        val_output=val_csv,
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        image_dir="geo_dataset/train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = GeoLocationModel().to(device)

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

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    os.makedirs(
        "checkpoints/cv",
        exist_ok=True,
    )

    if args.checkpoint_name is None:
        checkpoint_name = f"fold{args.fold}.pth"
    else:
        checkpoint_name = args.checkpoint_name

    checkpoint_path = os.path.join(
        "checkpoints/cv",
        checkpoint_name,
    )

    best_val_distance = float("inf")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            haversine_weight=args.haversine_weight,
        )

        val_median_km = validate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        scheduler.step(val_median_km)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Fold {args.fold} "
            f"| Epoch {epoch + 1}/{args.epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Median Haversine: {val_median_km:.2f} km "
            f"| LR: {current_lr:.6f}"
        )

        if val_median_km < best_val_distance:
            best_val_distance = val_median_km

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Saved new best model "
                f"({best_val_distance:.2f} km) "
                f"to {checkpoint_path}"
            )


if __name__ == "__main__":
    main()