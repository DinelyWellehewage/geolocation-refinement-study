import torch
import torch.nn as nn

from src.utils.metrics import median_haversine_km


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = loss_fn(predictions, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    return average_loss


def validate(model, dataloader, device):
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

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    median_distance = median_haversine_km(
        all_predictions,
        all_targets,
    )

    return median_distance.item()