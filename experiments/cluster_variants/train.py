import torch
import torch.nn as nn


def train_one_epoch_cluster(
    model,
    dataloader,
    optimizer,
    device,
):
    model.train()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, clusters in dataloader:
        images = images.to(device)
        clusters = clusters.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(
            logits,
            clusters,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == clusters
        ).sum().item()

        total += clusters.size(0)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
    }


def validate_cluster(
    model,
    dataloader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, clusters in dataloader:
            images = images.to(device)
            clusters = clusters.to(device)

            logits = model(images)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == clusters
            ).sum().item()

            total += clusters.size(0)

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                clusters.cpu()
            )

    all_predictions = torch.cat(
        all_predictions
    )

    all_targets = torch.cat(
        all_targets
    )

    return {
        "accuracy": correct / total,
        "predictions": all_predictions,
        "targets": all_targets,
    }