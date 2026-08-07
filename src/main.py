import os

import torch
from torch.optim import AdamW

from src.datasets.dataloaders import create_dataloaders
from src.models.model import GeoLocationModel
from src.training.train import train_one_epoch, validate


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader = create_dataloaders(
        train_csv="geo_dataset/train_split.csv",
        val_csv="geo_dataset/val_split.csv",
        image_dir="geo_dataset/train",
        batch_size=32,
        num_workers=2,
    )

    model = GeoLocationModel()
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    num_epochs = 5

    best_val_distance = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
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
            f"Epoch {epoch + 1}/{num_epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Median Haversine: {val_median_km:.2f} km"
        )

        if val_median_km < best_val_distance:
            best_val_distance = val_median_km

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth",
            )

            print(
                f"Saved new best model "
                f"({best_val_distance:.2f} km)"
            )


if __name__ == "__main__":
    main()