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
from src.utils.metrics import haversine_distance_km


def xyz_to_latlng(xyz):
    xyz = xyz / np.linalg.norm(
        xyz,
        axis=1,
        keepdims=True,
    )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lng = np.arctan2(y, x)
    hyp = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, hyp)

    return np.stack(
        [
            np.degrees(lat),
            np.degrees(lng),
        ],
        axis=1,
    )


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

    train_csv = (
        "geo_dataset/train_cluster64.csv"
    )

    val_csv = (
        "geo_dataset/val_cluster64.csv"
    )

    _, val_loader = (
        create_cluster_country_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
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

    all_cluster_predictions = []
    all_country_predictions = []

    with torch.no_grad():
        for (
            images,
            _,
            _,
        ) in val_loader:
            images = images.to(device)

            (
                cluster_logits,
                country_logits,
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

            all_cluster_predictions.append(
                cluster_predictions.cpu()
            )

            all_country_predictions.append(
                country_predictions.cpu()
            )

    predicted_clusters = torch.cat(
        all_cluster_predictions
    ).numpy()

    predicted_country_ids = torch.cat(
        all_country_predictions
    ).numpy()

    centers_xyz = np.load(
        "geo_dataset/"
        "cluster64_centers_xyz.npy"
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    val_df = pd.read_csv(
        val_csv
    )

    true_coordinates = torch.tensor(
        val_df[
            ["lat", "lng"]
        ].values,
        dtype=torch.float32,
    )

    predicted_coordinates = torch.tensor(
        centers_latlng[
            predicted_clusters
        ],
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    ).numpy()

    results = val_df.copy()

    results[
        "predicted_cluster"
    ] = predicted_clusters

    results[
        "predicted_country_id"
    ] = predicted_country_ids

    results[
        "distance_km"
    ] = distances

    rows = []

    for country_name, group in (
        results.groupby("country")
    ):
        country_distances = (
            group["distance_km"]
            .to_numpy()
        )

        rows.append(
            {
                "country": country_name,
                "samples": len(group),
                "median_km": float(
                    np.median(
                        country_distances
                    )
                ),
                "mean_km": float(
                    np.mean(
                        country_distances
                    )
                ),
                "within_200_pct": float(
                    (
                        country_distances
                        < 200
                    ).mean()
                    * 100
                ),
                "within_750_pct": float(
                    (
                        country_distances
                        < 750
                    ).mean()
                    * 100
                ),
                "catastrophic_pct": float(
                    (
                        country_distances
                        >= 750
                    ).mean()
                    * 100
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        "median_km"
    )

    print()
    print(
        "=== Performance by country ==="
    )
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    print()
    print("=== Overall ===")
    print(
        "Median:",
        f"{np.median(distances):.2f} km",
    )
    print(
        "Mean:",
        f"{np.mean(distances):.2f} km",
    )
    print(
        "<200:",
        f"{(distances < 200).mean() * 100:.2f}%",
    )
    print(
        "<750:",
        f"{(distances < 750).mean() * 100:.2f}%",
    )
    print(
        ">=750:",
        f"{(distances >= 750).mean() * 100:.2f}%",
    )

    output_path = (
        "geo_dataset/"
        "cluster_country_by_country.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()