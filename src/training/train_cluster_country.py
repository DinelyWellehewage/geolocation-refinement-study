import torch
import torch.nn as nn

from src.training.country_consistency import (
    CountryConsistencyLoss,
    build_cluster_country_matrix,
)
from src.training.geographic_label_smoothing import (
    GeographicLabelSmoothingLoss,
)


def build_cluster_loss(
    smoothing_mode,
    centers_xyz,
    label_smoothing,
    geographic_temperature_km,
):
    if smoothing_mode == "none":
        return nn.CrossEntropyLoss()

    if smoothing_mode == "ordinary":
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

    if smoothing_mode == "geographic":
        return GeographicLabelSmoothingLoss(
            centers_xyz=centers_xyz,
            smoothing=label_smoothing,
            temperature_km=geographic_temperature_km,
        )

    raise ValueError(
        f"Unknown smoothing mode: {smoothing_mode}"
    )


def build_consistency_loss(
    train_csv,
    num_clusters,
    num_countries,
    device,
):
    matrix = build_cluster_country_matrix(
        train_csv=train_csv,
        num_clusters=num_clusters,
        num_countries=num_countries,
    )

    loss_fn = CountryConsistencyLoss(
        cluster_country_matrix=matrix,
    )

    return loss_fn.to(device)


def train_one_epoch_cluster_country(
    model,
    dataloader,
    optimizer,
    device,
    country_weight=0.2,
    consistency_weight=0.0,
    consistency_loss_fn=None,
    smoothing_mode="ordinary",
    centers_xyz=None,
    label_smoothing=0.1,
    geographic_temperature_km=500.0,
):
    model.train()

    if smoothing_mode == "geographic":
        if centers_xyz is None:
            raise ValueError(
                "centers_xyz is required for "
                "geographic label smoothing."
            )

    if consistency_weight > 0:
        if consistency_loss_fn is None:
            raise ValueError(
                "consistency_loss_fn is required when "
                "consistency_weight > 0."
            )

    cluster_loss_fn = build_cluster_loss(
        smoothing_mode=smoothing_mode,
        centers_xyz=centers_xyz,
        label_smoothing=label_smoothing,
        geographic_temperature_km=(
            geographic_temperature_km
        ),
    )

    cluster_loss_fn = cluster_loss_fn.to(device)

    country_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cluster_loss = 0.0
    total_country_loss = 0.0
    total_consistency_loss = 0.0

    cluster_correct = 0
    country_correct = 0
    total = 0

    for images, clusters, countries in dataloader:
        images = images.to(device)
        clusters = clusters.to(device)
        countries = countries.to(device)

        optimizer.zero_grad()

        cluster_logits, country_logits = model(images)

        cluster_loss = cluster_loss_fn(
            cluster_logits,
            clusters,
        )

        country_loss = country_loss_fn(
            country_logits,
            countries,
        )

        if consistency_weight > 0:
            consistency_loss = consistency_loss_fn(
                cluster_logits,
                country_logits,
            )
        else:
            consistency_loss = torch.tensor(
                0.0,
                device=device,
            )

        loss = (
            cluster_loss
            + country_weight * country_loss
            + consistency_weight * consistency_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cluster_loss += cluster_loss.item()
        total_country_loss += country_loss.item()
        total_consistency_loss += (
            consistency_loss.item()
        )

        cluster_predictions = cluster_logits.argmax(
            dim=1
        )

        country_predictions = country_logits.argmax(
            dim=1
        )

        cluster_correct += (
            cluster_predictions == clusters
        ).sum().item()

        country_correct += (
            country_predictions == countries
        ).sum().item()

        total += clusters.size(0)

    num_batches = len(dataloader)

    return {
        "total_loss": (
            total_loss / num_batches
        ),
        "cluster_loss": (
            total_cluster_loss / num_batches
        ),
        "country_loss": (
            total_country_loss / num_batches
        ),
        "consistency_loss": (
            total_consistency_loss / num_batches
        ),
        "cluster_accuracy": (
            cluster_correct / total
        ),
        "country_accuracy": (
            country_correct / total
        ),
    }


def validate_cluster_country(
    model,
    dataloader,
    device,
):
    model.eval()

    cluster_correct = 0
    country_correct = 0
    total = 0

    all_cluster_predictions = []
    all_cluster_targets = []

    all_country_predictions = []
    all_country_targets = []

    with torch.no_grad():
        for images, clusters, countries in dataloader:
            images = images.to(device)
            clusters = clusters.to(device)
            countries = countries.to(device)

            cluster_logits, country_logits = model(images)

            cluster_predictions = cluster_logits.argmax(
                dim=1
            )

            country_predictions = country_logits.argmax(
                dim=1
            )

            cluster_correct += (
                cluster_predictions == clusters
            ).sum().item()

            country_correct += (
                country_predictions == countries
            ).sum().item()

            total += clusters.size(0)

            all_cluster_predictions.append(
                cluster_predictions.cpu()
            )

            all_cluster_targets.append(
                clusters.cpu()
            )

            all_country_predictions.append(
                country_predictions.cpu()
            )

            all_country_targets.append(
                countries.cpu()
            )

    return {
        "cluster_accuracy": (
            cluster_correct / total
        ),
        "country_accuracy": (
            country_correct / total
        ),
        "cluster_predictions": torch.cat(
            all_cluster_predictions
        ),
        "cluster_targets": torch.cat(
            all_cluster_targets
        ),
        "country_predictions": torch.cat(
            all_country_predictions
        ),
        "country_targets": torch.cat(
            all_country_targets
        ),
    }