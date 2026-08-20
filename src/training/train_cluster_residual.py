import torch
import torch.nn as nn


def train_one_epoch_cluster_residual(
    model,
    dataloader,
    optimizer,
    device,
    residual_weight=1.0,
):
    model.train()

    cluster_loss_fn = nn.CrossEntropyLoss()
    residual_loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_cluster_loss = 0.0
    total_residual_loss = 0.0

    correct = 0
    total = 0

    for images, clusters, residuals in dataloader:
        images = images.to(device)
        clusters = clusters.to(device)
        residuals = residuals.to(device)

        optimizer.zero_grad()

        cluster_logits, predicted_residuals = model(images)

        cluster_loss = cluster_loss_fn(
            cluster_logits,
            clusters,
        )

        residual_loss = residual_loss_fn(
            predicted_residuals,
            residuals,
        )

        loss = (
            cluster_loss
            + residual_weight * residual_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cluster_loss += cluster_loss.item()
        total_residual_loss += residual_loss.item()

        predictions = cluster_logits.argmax(dim=1)

        correct += (
            predictions == clusters
        ).sum().item()

        total += clusters.size(0)

    num_batches = len(dataloader)

    return {
        "total_loss": total_loss / num_batches,
        "cluster_loss": total_cluster_loss / num_batches,
        "residual_loss": total_residual_loss / num_batches,
        "cluster_accuracy": correct / total,
    }


def validate_cluster_residual(
    model,
    dataloader,
    device,
):
    model.eval()

    all_cluster_predictions = []
    all_cluster_targets = []

    all_residual_predictions = []
    all_residual_targets = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, clusters, residuals in dataloader:
            images = images.to(device)
            clusters = clusters.to(device)
            residuals = residuals.to(device)

            cluster_logits, predicted_residuals = model(images)

            cluster_predictions = cluster_logits.argmax(
                dim=1
            )

            correct += (
                cluster_predictions == clusters
            ).sum().item()

            total += clusters.size(0)

            all_cluster_predictions.append(
                cluster_predictions.cpu()
            )

            all_cluster_targets.append(
                clusters.cpu()
            )

            all_residual_predictions.append(
                predicted_residuals.cpu()
            )

            all_residual_targets.append(
                residuals.cpu()
            )

    return {
        "cluster_accuracy": correct / total,
        "cluster_predictions": torch.cat(
            all_cluster_predictions
        ),
        "cluster_targets": torch.cat(
            all_cluster_targets
        ),
        "residual_predictions": torch.cat(
            all_residual_predictions
        ),
        "residual_targets": torch.cat(
            all_residual_targets
        ),
    }