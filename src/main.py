import argparse
import os

import torch
from torch.optim import AdamW

from src.datasets.dataloaders import create_dataloaders
from src.models.model import GeoLocationModel
from src.training.train import train_one_epoch, validate

import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pth",
    )
    parser.add_argument("--seed", type=int, default=42)

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
    print("Random seed:", args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader = create_dataloaders(
        train_csv="geo_dataset/train_split.csv",
        val_csv="geo_dataset/val_split.csv",
        image_dir="geo_dataset/train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = GeoLocationModel().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    best_val_distance = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = os.path.join(
        "checkpoints",
        args.checkpoint_name,
    )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_median_km = validate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Median Haversine: {val_median_km:.2f} km"
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