import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)

from experiments.regression_pretrained.dataloaders import (
    create_regression_pretrained_dataloaders,
)
from experiments.regression_pretrained.model import (
    GeoRegressionPretrainedModel,
)
from experiments.regression_pretrained.train import (
    train_one_epoch,
)
from src.utils.metrics import (
    denormalize_coordinates,
    haversine_distance_km,
)


def parse_args():
    parser = argparse.ArgumentParser()

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
        default=(
            "regression_imagenet_seed42.pth"
        ),
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_full(
    model,
    dataloader,
    device,
):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for (
            images,
            targets,
        ) in dataloader:
            images = images.to(
                device
            )

            targets = targets.to(
                device
            )

            predictions = model(
                images
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    predictions = (
        denormalize_coordinates(
            predictions
        )
    )

    targets = (
        denormalize_coordinates(
            targets
        )
    )

    distances = (
        haversine_distance_km(
            predictions,
            targets,
        )
    )

    median_km = torch.median(
        distances
    ).item()

    mean_km = torch.mean(
        distances
    ).item()

    within_200 = (
        (distances < 200)
        .float()
        .mean()
        .item()
        * 100
    )

    within_750 = (
        (distances < 750)
        .float()
        .mean()
        .item()
        * 100
    )

    return {
        "median_km": median_km,
        "mean_km": mean_km,
        "within_200": within_200,
        "within_750": within_750,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Model: ImageNet-pretrained "
        "MobileNetV3-Large regression"
    )

    print(
        "Using device:",
        device,
    )

    train_loader, val_loader = (
        create_regression_pretrained_dataloaders(
            train_csv=(
                "geo_dataset/train_split.csv"
            ),
            val_csv=(
                "geo_dataset/val_split.csv"
            ),
            image_dir=(
                "geo_dataset/train"
            ),
            batch_size=(
                args.batch_size
            ),
            num_workers=(
                args.num_workers
            ),
            seed=args.seed,
        )
    )

    model = (
        GeoRegressionPretrainedModel()
        .to(device)
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameter count:",
        num_params,
    )

    print(
        "Under 5 million:",
        num_params <= 5_000_000,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
    )

    checkpoint_dir = (
        "checkpoints/regression_pretrained"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        args.checkpoint_name,
    )

    best_median = float(
        "inf"
    )

    for epoch in range(
        args.epochs
    ):
        train_loss = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
            )
        )

        metrics = validate_full(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        scheduler.step(
            metrics["median_km"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Train Loss: "
            f"{train_loss:.4f} "
            f"| Median: "
            f"{metrics['median_km']:.2f} km "
            f"| Mean: "
            f"{metrics['mean_km']:.2f} km "
            f"| <200: "
            f"{metrics['within_200']:.2f}% "
            f"| <750: "
            f"{metrics['within_750']:.2f}% "
            f"| LR: "
            f"{current_lr:.6f}"
        )

        if (
            metrics["median_km"]
            < best_median
        ):
            best_median = (
                metrics[
                    "median_km"
                ]
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                "Saved new best "
                "pretrained regression model "
                f"({best_median:.2f} km) "
                f"to {checkpoint_path}"
            )

    print()
    print(
        "Best validation median:",
        f"{best_median:.2f} km",
    )


if __name__ == "__main__":
    main()