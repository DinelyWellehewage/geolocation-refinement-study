import torch
import torch.nn as nn

from src.training.country_consistency import (
    CountryConsistencyLoss,
)


def train_one_epoch_cluster_country_residual(
    model,
    dataloader,
    optimizer,
    device,
    cluster_loss_fn,
    consistency_loss_fn,
    country_weight=0.1,
    consistency_weight=0.05,
    residual_weight=1.0,
):
    model.train()

    country_loss_fn = nn.CrossEntropyLoss()
    residual_loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_cluster_loss = 0.0
    total_country_loss = 0.0
    total_consistency_loss = 0.0
    total_residual_loss = 0.0

    cluster_correct = 0
    country_correct = 0
    total_samples = 0

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

        consistency_loss = consistency_loss_fn(
            cluster_logits,
            country_logits,
        )

        residual_loss = residual_loss_fn(
            residual_predictions,
            residual_targets,
        )

        loss = (
            cluster_loss
            + country_weight * country_loss
            + consistency_weight * consistency_loss
            + residual_weight * residual_loss
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_samples += batch_size

        total_loss += (
            loss.item()
            * batch_size
        )

        total_cluster_loss += (
            cluster_loss.item()
            * batch_size
        )

        total_country_loss += (
            country_loss.item()
            * batch_size
        )

        total_consistency_loss += (
            consistency_loss.item()
            * batch_size
        )

        total_residual_loss += (
            residual_loss.item()
            * batch_size
        )

        cluster_predictions = (
            cluster_logits.argmax(
                dim=1
            )
        )

        country_predictions = (
            country_logits.argmax(
                dim=1
            )
        )

        cluster_correct += (
            cluster_predictions
            == clusters
        ).sum().item()

        country_correct += (
            country_predictions
            == countries
        ).sum().item()

    return {
        "total_loss": (
            total_loss
            / total_samples
        ),
        "cluster_loss": (
            total_cluster_loss
            / total_samples
        ),
        "country_loss": (
            total_country_loss
            / total_samples
        ),
        "consistency_loss": (
            total_consistency_loss
            / total_samples
        ),
        "residual_loss": (
            total_residual_loss
            / total_samples
        ),
        "cluster_accuracy": (
            cluster_correct
            / total_samples
        ),
        "country_accuracy": (
            country_correct
            / total_samples
        ),
    }


def validate_cluster_country_residual(
    model,
    dataloader,
    device,
    centers_xyz,
):
    model.eval()

    cluster_correct = 0
    country_correct = 0
    total_samples = 0

    all_cluster_predictions = []
    all_country_predictions = []
    all_residual_predictions = []
    all_residual_targets = []
    all_predicted_xyz = []

    centers_tensor = torch.tensor(
        centers_xyz,
        dtype=torch.float32,
        device=device,
    )

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
            residual_targets = residual_targets.to(device)

            (
                cluster_logits,
                country_logits,
                residual_predictions,
            ) = model(images)

            cluster_predictions = (
                cluster_logits.argmax(
                    dim=1
                )
            )

            country_predictions = (
                country_logits.argmax(
                    dim=1
                )
            )

            predicted_centers = (
                centers_tensor[
                    cluster_predictions
                ]
            )

            predicted_xyz = (
                predicted_centers
                + residual_predictions
            )

            predicted_xyz = (
                predicted_xyz
                / predicted_xyz.norm(
                    dim=1,
                    keepdim=True,
                ).clamp_min(
                    1e-12
                )
            )

            batch_size = images.size(0)

            total_samples += batch_size

            cluster_correct += (
                cluster_predictions
                == clusters
            ).sum().item()

            country_correct += (
                country_predictions
                == countries
            ).sum().item()

            all_cluster_predictions.append(
                cluster_predictions.cpu()
            )

            all_country_predictions.append(
                country_predictions.cpu()
            )

            all_residual_predictions.append(
                residual_predictions.cpu()
            )

            all_residual_targets.append(
                residual_targets.cpu()
            )

            all_predicted_xyz.append(
                predicted_xyz.cpu()
            )

    return {
        "cluster_accuracy": (
            cluster_correct
            / total_samples
        ),
        "country_accuracy": (
            country_correct
            / total_samples
        ),
        "cluster_predictions": torch.cat(
            all_cluster_predictions
        ),
        "country_predictions": torch.cat(
            all_country_predictions
        ),
        "residual_predictions": torch.cat(
            all_residual_predictions
        ),
        "residual_targets": torch.cat(
            all_residual_targets
        ),
        "predicted_xyz": torch.cat(
            all_predicted_xyz
        ),
    }