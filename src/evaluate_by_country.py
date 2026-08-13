import pandas as pd
import torch

import argparse

from src.datasets.dataloaders import create_dataloaders
from src.models.model import GeoLocationModel
from src.utils.metrics import (
    denormalize_coordinates,
    haversine_distance_km,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    _, val_loader = create_dataloaders(
        train_csv="geo_dataset/train_split.csv",
        val_csv="geo_dataset/val_split.csv",
        image_dir="geo_dataset/train",
        batch_size=32,
        num_workers=2,
    )

    model = GeoLocationModel()

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
        )
    )

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            predictions = model(images)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets)

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    predictions = denormalize_coordinates(predictions)
    targets = denormalize_coordinates(targets)

    distances = haversine_distance_km(
        predictions,
        targets,
    ).numpy()

    val_df = pd.read_csv(
        "geo_dataset/val_split.csv"
    ).reset_index(drop=True)

    val_df["distance_km"] = distances

    country_results = (
        val_df
        .groupby("country")["distance_km"]
        .agg(
            count="count",
            median_km="median",
            mean_km="mean",
        )
        .sort_values("median_km")
    )

    print()
    print(country_results.round(2))


if __name__ == "__main__":
    main()
