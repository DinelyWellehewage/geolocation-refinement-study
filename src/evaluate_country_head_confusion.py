import argparse

import numpy as np
import pandas as pd
import torch

from src.datasets.dataloaders_cluster_country import (
    create_cluster_country_dataloaders,
)
from src.models.model_cluster_country import (
    GeoClusterCountryModel,
)


COUNTRY_NAMES = [
    "Belarus",
    "Finland",
    "France",
    "Germany",
    "Iceland",
    "Italy",
    "Norway",
    "Poland",
    "Spain",
    "Sweden",
    "Turkey",
    "United_Kingdom",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-countries",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)

    _, val_loader = (
        create_cluster_country_dataloaders(
            train_csv="geo_dataset/train_cluster64.csv",
            val_csv="geo_dataset/val_cluster64.csv",
            image_dir="geo_dataset/train",
            batch_size=args.batch_size,
            num_workers=0,
            seed=42,
        )
    )

    model = GeoClusterCountryModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
    ).to(device)

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for (
            images,
            _,
            countries,
        ) in val_loader:
            images = images.to(device)

            _, country_logits = model(
                images
            )

            predictions = (
                country_logits.argmax(
                    dim=1
                )
            )

            all_true.append(
                countries.cpu()
            )

            all_pred.append(
                predictions.cpu()
            )

    true_ids = torch.cat(
        all_true
    ).numpy()

    pred_ids = torch.cat(
        all_pred
    ).numpy()

    confusion = np.zeros(
        (
            args.num_countries,
            args.num_countries,
        ),
        dtype=np.int64,
    )

    for true_id, pred_id in zip(
        true_ids,
        pred_ids,
    ):
        confusion[
            true_id,
            pred_id,
        ] += 1

    total_correct = np.trace(
        confusion
    )

    total = confusion.sum()

    overall_accuracy = (
        total_correct
        / total
        * 100
    )

    print()
    print(
        "Overall country-head accuracy:",
        f"{overall_accuracy:.2f}%",
    )

    rows = []

    for country_id in range(
        args.num_countries
    ):
        row = confusion[
            country_id
        ]

        samples = row.sum()

        correct = row[
            country_id
        ]

        accuracy = (
            correct
            / samples
            * 100
            if samples > 0
            else 0.0
        )

        wrong_row = row.copy()

        wrong_row[
            country_id
        ] = 0

        top_wrong_id = int(
            np.argmax(
                wrong_row
            )
        )

        top_wrong_count = int(
            wrong_row[
                top_wrong_id
            ]
        )

        rows.append(
            {
                "country": (
                    COUNTRY_NAMES[
                        country_id
                    ]
                ),
                "samples": int(
                    samples
                ),
                "correct": int(
                    correct
                ),
                "accuracy_pct": float(
                    accuracy
                ),
                "top_wrong_country": (
                    COUNTRY_NAMES[
                        top_wrong_id
                    ]
                    if top_wrong_count > 0
                    else ""
                ),
                "top_wrong_count": (
                    top_wrong_count
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        "accuracy_pct"
    )

    print()
    print(
        "=== Country-head accuracy by country ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=COUNTRY_NAMES,
        columns=COUNTRY_NAMES,
    )

    print()
    print(
        "=== Country-head confusion matrix ==="
    )

    print(
        confusion_df.to_string()
    )

    summary_path = (
        "geo_dataset/"
        "country_head_accuracy.csv"
    )

    confusion_path = (
        "geo_dataset/"
        "country_head_confusion.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    confusion_df.to_csv(
        confusion_path,
    )

    print()
    print("Saved:")
    print(summary_path)
    print(confusion_path)


if __name__ == "__main__":
    main()