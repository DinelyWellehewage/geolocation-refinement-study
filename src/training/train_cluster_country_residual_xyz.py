import torch
import torch.nn as nn


def train_one_epoch_cluster_country_residual_xyz(
    model,
    dataloader,
    optimizer,
    device,
    country_weight=0.2,
    residual_weight=1.0,
    label_smoothing=0.1,
):
    model.train()

    cluster_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )

    country_loss_fn = nn.CrossEntropyLoss()

    residual_loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_cluster_loss = 0.0
    total_country_loss = 0.0
    total_residual_loss = 0.0

    cluster_correct = 0
    country_correct = 0
    total = 0

    for (
        images,
        clusters,
        countries,
        residual_targets,
    ) in dataloader:
        images = images.to(device)
        clusters = clusters.to(device)
        countries = countries.to(device)
        residual_targets = residual_targets.to(device)

        optimizer.zero_grad()

        (
            cluster_logits,
            country_logits,
            residual_predictions,
        ) = model(images)

        cluster_loss = cluster_loss_fn(
            cluster_logits,
            clusters,
        )

        country_loss = country_loss_fn(
            country_logits,
            countries,
        )

        residual_loss = residual_loss_fn(
            residual_predictions,
            residual_targets,
        )

        loss = (
            cluster_loss
            + country_weight * country_loss
            + residual_weight * residual_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cluster_loss += cluster_loss.item()
        total_country_loss += country_loss.item()
        total_residual_loss += residual_loss.item()

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
        "residual_loss": (
            total_residual_loss / num_batches
        ),
        "cluster_accuracy": (
            cluster_correct / total
        ),
        "country_accuracy": (
            country_correct / total
        ),
    }


def validate_cluster_country_residual_xyz(
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

    all_residual_predictions = []
    all_residual_targets = []

    with torch.no_grad():
        for (
            images,
            clusters,
            countries,
            residual_targets,
        ) in dataloader:
            images = images.to(device)
            clusters = clusters.to(device)
            countries = countries.to(device)

            (
                cluster_logits,
                country_logits,
                residual_predictions,
            ) = model(images)

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

            all_residual_predictions.append(
                residual_predictions.cpu()
            )

            all_residual_targets.append(
                residual_targets.cpu()
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
        "residual_predictions": torch.cat(
            all_residual_predictions
        ),
        "residual_targets": torch.cat(
            all_residual_targets
        ),
    }
