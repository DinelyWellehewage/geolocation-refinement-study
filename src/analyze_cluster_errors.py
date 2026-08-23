import argparse

import numpy as np
import pandas as pd
import torch

from src.datasets.dataloaders_cluster import (
    create_cluster_dataloaders,
)
from src.datasets.dataloaders_cluster_country import (
    create_cluster_country_dataloaders,
)
from src.models.model_cluster import (
    GeoClusterModel,
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


def build_model(
    model_type,
    num_clusters,
    num_countries,
    pretrained,
):
    if model_type == "cluster":
        if pretrained:
            raise ValueError(
                "--pretrained is currently supported only "
                "for model-type=cluster_country."
            )

        return GeoClusterModel(
            num_clusters=num_clusters,
        )

    if model_type == "cluster_country":
        return GeoClusterCountryModel(
            num_clusters=num_clusters,
            num_countries=num_countries,
            pretrained=pretrained,
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=[
            "cluster",
            "cluster_country",
        ],
        default="cluster",
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

    parser.add_argument(
        "--pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    k = args.num_clusters

    if args.output_prefix is None:
        args.output_prefix = (
            f"cluster{k}"
        )

    train_csv = (
        f"geo_dataset/"
        f"train_cluster{k}.csv"
    )

    val_csv = (
        f"geo_dataset/"
        f"val_cluster{k}.csv"
    )

    centers_path = (
        f"geo_dataset/"
        f"cluster{k}_centers_xyz.npy"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Model type:", args.model_type)
    print("Number of clusters:", k)
    print(
        "Pretrained preprocessing:",
        args.pretrained,
    )

    if args.model_type == "cluster_country":
        _, val_loader = (
            create_cluster_country_dataloaders(
                train_csv=train_csv,
                val_csv=val_csv,
                image_dir="geo_dataset/train",
                batch_size=args.batch_size,
                num_workers=0,
                seed=42,
                pretrained=args.pretrained,
            )
        )
    else:
        _, val_loader = (
            create_cluster_dataloaders(
                train_csv=train_csv,
                val_csv=val_csv,
                image_dir="geo_dataset/train",
                batch_size=args.batch_size,
                num_workers=0,
                seed=42,
            )
        )

    model = build_model(
        model_type=args.model_type,
        num_clusters=k,
        num_countries=args.num_countries,
        pretrained=args.pretrained,
    ).to(device)

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        if args.model_type == "cluster_country":
            for (
                images,
                clusters,
                _,
            ) in val_loader:
                images = images.to(
                    device
                )

                (
                    cluster_logits,
                    _,
                ) = model(images)

                predictions = (
                    cluster_logits.argmax(
                        dim=1
                    )
                )

                all_predictions.append(
                    predictions.cpu()
                )

                all_targets.append(
                    clusters.cpu()
                )

        else:
            for (
                images,
                clusters,
            ) in val_loader:
                images = images.to(
                    device
                )

                logits = model(
                    images
                )

                predictions = logits.argmax(
                    dim=1
                )

                all_predictions.append(
                    predictions.cpu()
                )

                all_targets.append(
                    clusters.cpu()
                )

    predicted_clusters = torch.cat(
        all_predictions
    ).numpy()

    true_clusters = torch.cat(
        all_targets
    ).numpy()

    centers_xyz = np.load(
        centers_path
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    val_df = pd.read_csv(
        val_csv
    )

    true_coordinates = torch.tensor(
        val_df[
            [
                "lat",
                "lng",
            ]
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

    correct_cluster = (
        predicted_clusters
        == true_clusters
    )

    wrong_cluster = (
        ~correct_cluster
    )

    near_wrong = (
        wrong_cluster
        & (distances < 200)
    )

    medium_wrong = (
        wrong_cluster
        & (distances >= 200)
        & (distances < 750)
    )

    catastrophic_wrong = (
        wrong_cluster
        & (distances >= 750)
    )

    total = len(
        distances
    )

    def pct(mask):
        return (
            mask.sum()
            / total
            * 100
        )

    print()
    print(
        "=== Error breakdown ==="
    )

    print(
        "Correct cluster:",
        correct_cluster.sum(),
        f"({pct(correct_cluster):.2f}%)",
    )

    print(
        "Wrong cluster but <200 km:",
        near_wrong.sum(),
        f"({pct(near_wrong):.2f}%)",
    )

    print(
        "Wrong cluster 200-750 km:",
        medium_wrong.sum(),
        f"({pct(medium_wrong):.2f}%)",
    )

    print(
        "Wrong cluster >=750 km:",
        catastrophic_wrong.sum(),
        f"({pct(catastrophic_wrong):.2f}%)",
    )

    print()
    print(
        "=== Distance metrics ==="
    )

    print(
        "Median:",
        f"{np.median(distances):.2f} km",
    )

    print(
        "Mean:",
        f"{np.mean(distances):.2f} km",
    )

    print(
        "<200 km:",
        (
            f"{(distances < 200).mean() * 100:.2f}%"
        ),
    )

    print(
        "<750 km:",
        (
            f"{(distances < 750).mean() * 100:.2f}%"
        ),
    )

    confusion = np.zeros(
        (
            k,
            k,
        ),
        dtype=np.int64,
    )

    for true_id, pred_id in zip(
        true_clusters,
        predicted_clusters,
    ):
        confusion[
            true_id,
            pred_id,
        ] += 1

    error_df = val_df.copy()

    error_df[
        "true_cluster"
    ] = true_clusters

    error_df[
        "predicted_cluster"
    ] = predicted_clusters

    error_df[
        "distance_km"
    ] = distances

    error_df[
        "correct_cluster"
    ] = correct_cluster

    error_df[
        "catastrophic"
    ] = catastrophic_wrong

    confusion_path = (
        f"geo_dataset/"
        f"{args.output_prefix}_"
        "confusion.npy"
    )

    error_path = (
        f"geo_dataset/"
        f"{args.output_prefix}_"
        "error_analysis.csv"
    )

    np.save(
        confusion_path,
        confusion,
    )

    error_df.to_csv(
        error_path,
        index=False,
    )

    catastrophic_df = error_df[
        error_df["catastrophic"]
    ].copy()

    country_pairs_path = None

    if len(
        catastrophic_df
    ) > 0:
        train_df = pd.read_csv(
            train_csv
        )

        cluster_country_table = (
            pd.crosstab(
                train_df[
                    "geo_cluster"
                ],
                train_df[
                    "country"
                ],
            )
        )

        cluster_to_country = (
            cluster_country_table
            .idxmax(
                axis=1
            )
            .to_dict()
        )

        catastrophic_df[
            "predicted_country"
        ] = (
            catastrophic_df[
                "predicted_cluster"
            ]
            .map(
                cluster_to_country
            )
        )

        same_country = (
            catastrophic_df[
                "country"
            ]
            == catastrophic_df[
                "predicted_country"
            ]
        )

        same_count = int(
            same_country.sum()
        )

        cross_count = int(
            (
                ~same_country
            ).sum()
        )

        print()
        print(
            "=== Catastrophic error "
            "country analysis ==="
        )

        print(
            "Catastrophic same-country:",
            same_count,
            (
                f"({same_count / len(catastrophic_df) * 100:.2f}% "
                "of catastrophic errors)"
            ),
        )

        print(
            "Catastrophic cross-country:",
            cross_count,
            (
                f"({cross_count / len(catastrophic_df) * 100:.2f}% "
                "of catastrophic errors)"
            ),
        )

        country_pairs = (
            catastrophic_df
            .groupby(
                [
                    "country",
                    "predicted_country",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
        )

        country_pairs = (
            country_pairs.rename(
                columns={
                    "country": (
                        "true_country"
                    ),
                }
            )
        )

        print()
        print(
            "=== Top catastrophic "
            "country pairs ==="
        )

        print(
            country_pairs
            .head(20)
            .to_string(
                index=False
            )
        )

        country_pairs_path = (
            f"geo_dataset/"
            f"{args.output_prefix}_"
            "catastrophic_country_pairs.csv"
        )

        country_pairs.to_csv(
            country_pairs_path,
            index=False,
        )

    print()
    print("Saved:")
    print(
        confusion_path
    )
    print(
        error_path
    )

    if (
        country_pairs_path
        is not None
    ):
        print(
            country_pairs_path
        )


if __name__ == "__main__":
    main()