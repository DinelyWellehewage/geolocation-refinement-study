import torch
import torch.nn as nn

from src.utils.metrics import (
    denormalize_coordinates,
    haversine_distance_km,
    median_haversine_km,
)


def coordinate_loss(
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


def train_one_epoch_multitask(
    model,
    dataloader,
    optimizer,
    device,
    haversine_weight=0.1,
    country_weight=0.2,
):
    model.train()

    country_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_coordinate_loss = 0.0
    total_country_loss = 0.0

    for images, coordinates, countries in dataloader:
        images = images.to(device)
        coordinates = coordinates.to(device)
        countries = countries.to(device)

        optimizer.zero_grad()

        predicted_coordinates, country_logits = model(images)

        coord_loss = coordinate_loss(
            predicted_coordinates,
            coordinates,
            haversine_weight=haversine_weight,
        )

        country_loss = country_loss_fn(
            country_logits,
            countries,
        )

        loss = (
            coord_loss
            + country_weight * country_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_coordinate_loss += coord_loss.item()
        total_country_loss += country_loss.item()

    num_batches = len(dataloader)

    return {
        "total_loss": total_loss / num_batches,
        "coordinate_loss": total_coordinate_loss / num_batches,
        "country_loss": total_country_loss / num_batches,
    }


def validate_multitask(
    model,
    dataloader,
    device,
):
    model.eval()

    all_predictions = []
    all_targets = []

    correct_countries = 0
    total_countries = 0

    with torch.no_grad():
        for images, coordinates, countries in dataloader:
            images = images.to(device)
            coordinates = coordinates.to(device)
            countries = countries.to(device)

            predicted_coordinates, country_logits = model(images)

            all_predictions.append(
                predicted_coordinates.cpu()
            )

            all_targets.append(
                coordinates.cpu()
            )

            predicted_countries = country_logits.argmax(
                dim=1
            )

            correct_countries += (
                predicted_countries == countries
            ).sum().item()

            total_countries += countries.size(0)

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

    country_accuracy = (
        correct_countries / total_countries
    )

    return {
        "median_haversine_km": median_distance.item(),
        "country_accuracy": country_accuracy,
    }