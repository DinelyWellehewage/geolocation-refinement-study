import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from experiments.regression_pretrained.dataset import GeoDataset

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_regression_pretrained_dataloaders(
    train_csv,
    val_csv,
    image_dir,
    batch_size=64,
    num_workers=4,
    seed=42,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(
                p=0.05,
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.15),
            ),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    train_dataset = GeoDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=train_transform,
    )

    val_dataset = GeoDataset(
        csv_file=val_csv,
        image_dir=image_dir,
        transform=val_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    return (
        train_loader,
        val_loader,
    )