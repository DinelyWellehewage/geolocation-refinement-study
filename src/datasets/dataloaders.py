from torch.utils.data import DataLoader

from src.datasets.dataset import GeoDataset


def create_dataloaders(
    train_csv,
    val_csv,
    image_dir,
    batch_size=32,
    num_workers=2,
):
    train_dataset = GeoDataset(
        csv_file=train_csv,
        image_dir=image_dir,
    )

    val_dataset = GeoDataset(
        csv_file=val_csv,
        image_dir=image_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader