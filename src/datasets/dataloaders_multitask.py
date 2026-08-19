import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.dataset_multitask import MultiTaskGeoDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_multitask_dataloaders(
    train_csv,
    val_csv,
    image_dir,
    batch_size=32,
    num_workers=2,
    seed=42,
):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = MultiTaskGeoDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=train_transform,
    )

    val_dataset = MultiTaskGeoDataset(
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
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader