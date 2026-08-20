import torch
import torch.nn as nn


def train_one_epoch_cluster_country(
    model,
    dataloader,
    optimizer,
    device,
    country_weight=0.2,
):
    model.train()

    cluster_loss_fn = nn.CrossEntropyLoss()
    country_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cluster_loss = 0.0
    total_country_loss = 0.0

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

        loss = (
            cluster_loss
            + country_weight * country_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cluster_loss += cluster_loss.item()
        total_country_loss += country_loss.item()

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
        "total_loss": total_loss / num_batches,
        "cluster_loss": total_cluster_loss / num_batches,
        "country_loss": total_country_loss / num_batches,
        "cluster_accuracy": cluster_correct / total,
        "country_accuracy": country_correct / total,
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
        "cluster_accuracy": cluster_correct / total,
        "country_accuracy": country_correct / total,
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