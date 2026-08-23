import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW

from src.datasets.dataloaders_cluster_country import (
    create_cluster_country_dataloaders,
)
from src.models.model_cluster_country import (
    GeoClusterCountryModel,
)
from src.training.train_cluster_country import (
    train_one_epoch_cluster_country,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-countries",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--country-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final_k128_imagenet_cw020.pth",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Final full-data training")
    print("Using device:", device)
    print("Clusters:", args.num_clusters)
    print("Countries:", args.num_countries)
    print("Pretrained: True")
    print("Country weight:", args.country_weight)
    print("Weight decay:", args.weight_decay)
    print("Label smoothing:", args.label_smoothing)
    print("Epochs:", args.epochs)

    train_csv = (
        "geo_dataset/final/"
        f"train_cluster{args.num_clusters}_full.csv"
    )

    centers_path = (
        "geo_dataset/final/"
        f"cluster{args.num_clusters}_centers_full.npy"
    )

    # We only need a train loader.
    # Reuse the same CSV for val_csv, but ignore the returned val loader.
    train_loader, _ = (
        create_cluster_country_dataloaders(
            train_csv=train_csv,
            val_csv=train_csv,
            image_dir="geo_dataset/train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pretrained=True,
        )
    )

    model = GeoClusterCountryModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
        pretrained=True,
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameter count:", num_params)
    print(
        "Under 5 million:",
        num_params <= 5_000_000,
    )

    centers_xyz = np.load(
        centers_path
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = (
        "checkpoints/final"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        args.checkpoint_name,
    )

    for epoch in range(
        args.epochs
    ):
        train_metrics = (
            train_one_epoch_cluster_country(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                country_weight=args.country_weight,
                consistency_weight=0.0,
                consistency_loss_fn=None,
                smoothing_mode="ordinary",
                centers_xyz=centers_xyz,
                label_smoothing=args.label_smoothing,
                geographic_temperature_km=500.0,
            )
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Total Loss: "
            f"{train_metrics['total_loss']:.4f} "
            f"| Cluster Loss: "
            f"{train_metrics['cluster_loss']:.4f} "
            f"| Country Loss: "
            f"{train_metrics['country_loss']:.4f} "
            f"| Train Cluster Acc: "
            f"{train_metrics['cluster_accuracy'] * 100:.2f}% "
            f"| Train Country Acc: "
            f"{train_metrics['country_accuracy'] * 100:.2f}%"
        )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print()
    print("Final model saved:")
    print(checkpoint_path)


if __name__ == "__main__":
    main()