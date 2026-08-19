import torch
import torch.nn as nn

from src.utils.metrics import (
    denormalize_coordinates,
    haversine_distance_km,
    median_haversine_km,
)


def geolocation_loss(
    predictions,
    targets,
    haversine_weight=0.1,
):
    mse_loss = nn.functional.mse_loss(
        predictions,
        targets,
    )

    predictions_real = denormalize_coordinates(predictions)
    targets_real = denormalize_coordinates(targets)

    haversine_loss = haversine_distance_km(
        predictions_real,
        targets_real,
    ).mean()

    haversine_loss = haversine_loss / 1000.0

    return mse_loss + haversine_weight * haversine_loss


def train_one_epoch_cv(
    model,
    dataloader,
    optimizer,
    device,
    haversine_weight=0.1,
):
    model.train()

    total_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = geolocation_loss(
            predictions,
            targets,
            haversine_weight=haversine_weight,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_cv(
    model,
    dataloader,
    device,
):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    all_predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    all_targets = torch.cat(
        all_targets,
        dim=0,
    )

    all_predictions = denormalize_coordinates(
        all_predictions
    )

    all_targets = denormalize_coordinates(
        all_targets
    )

    median_distance = median_haversine_km(
        all_predictions,
        all_targets,
    )

    return median_distance.item()