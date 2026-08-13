import torch
import argparse
from src.datasets.dataloaders import create_dataloaders
from src.models.model import GeoLocationModel
from src.utils.metrics import (
    denormalize_coordinates,
    haversine_distance_km,
)


def evaluate_model(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            predictions = model(images)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_predictions = denormalize_coordinates(all_predictions)
    all_targets = denormalize_coordinates(all_targets)

    distances = haversine_distance_km(
        all_predictions,
        all_targets,
    )

    median_distance = torch.median(distances).item()
    mean_distance = torch.mean(distances).item()

    within_200 = (distances < 200).float().mean().item() * 100
    within_750 = (distances < 750).float().mean().item() * 100

    print(f"Median Haversine: {median_distance:.2f} km")
    print(f"Mean Haversine:   {mean_distance:.2f} km")
    print(f"Within 200 km:    {within_200:.2f}%")
    print(f"Within 750 km:    {within_750:.2f}%")

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
    print("Checkpoint:", args.checkpoint)

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

    evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
    )


if __name__ == "__main__":
    main()