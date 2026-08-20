import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.datasets.dataloaders_cluster_country import (
    create_cluster_country_dataloaders,
)
from src.models.model_cluster_country import (
    GeoClusterCountryModel,
)
from src.utils.metrics import haversine_distance_km


COUNTRY_TO_INDEX = {
    "Belarus": 0,
    "Finland": 1,
    "France": 2,
    "Germany": 3,
    "Iceland": 4,
    "Italy": 5,
    "Norway": 6,
    "Poland": 7,
    "Spain": 8,
    "Sweden": 9,
    "Turkey": 10,
    "United_Kingdom": 11,
}


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


def build_cluster_country_matrix(
    train_df,
    num_clusters,
    num_countries,
):
    matrix = np.zeros(
        (
            num_clusters,
            num_countries,
        ),
        dtype=np.float32,
    )

    for cluster_id in range(
        num_clusters
    ):
        cluster_rows = train_df[
            train_df["geo_cluster"]
            == cluster_id
        ]

        if len(cluster_rows) == 0:
            matrix[
                cluster_id
            ] = (
                1.0
                / num_countries
            )

            continue

        counts = (
            cluster_rows["country"]
            .value_counts()
        )

        for country_name, count in counts.items():
            country_index = COUNTRY_TO_INDEX[
                country_name
            ]

            matrix[
                cluster_id,
                country_index,
            ] = count

        row_sum = matrix[
            cluster_id
        ].sum()

        if row_sum > 0:
            matrix[
                cluster_id
            ] /= row_sum

    return matrix


def evaluate_alpha(
    cluster_log_probs,
    country_log_probs,
    cluster_country_probs,
    centers_latlng,
    true_coordinates,
    alpha,
):
    cluster_country_probs = np.clip(
        cluster_country_probs,
        1e-8,
        1.0,
    )

    cluster_country_log_probs = np.log(
        cluster_country_probs
    )

    # For every sample and every cluster:
    #
    # country score(cluster)
    # =
    # sum_c [
    #     P(country=c | cluster)
    #     * log P(country=c | image)
    # ]
    country_scores = (
        country_log_probs
        @ cluster_country_log_probs.T
    )

    fused_scores = (
        cluster_log_probs
        + alpha * country_scores
    )

    predicted_clusters = np.argmax(
        fused_scores,
        axis=1,
    )

    predicted_coordinates = centers_latlng[
        predicted_clusters
    ]

    predicted_coordinates = torch.tensor(
        predicted_coordinates,
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    ).numpy()

    return {
        "alpha": alpha,
        "median_km": float(
            np.median(distances)
        ),
        "mean_km": float(
            np.mean(distances)
        ),
        "within_200": float(
            (distances < 200).mean()
            * 100
        ),
        "within_750": float(
            (distances < 750).mean()
            * 100
        ),
        "catastrophic": float(
            (distances >= 750).mean()
            * 100
        ),
        "predicted_clusters": (
            predicted_clusters
        ),
        "distances": distances,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/cluster_country/"
            "cluster64_country_ls01_seed42.pth"
        ),
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

    train_df = pd.read_csv(
        "geo_dataset/train_cluster64.csv"
    )

    val_df = pd.read_csv(
        "geo_dataset/val_cluster64.csv"
    )

    cluster_country_probs = (
        build_cluster_country_matrix(
            train_df=train_df,
            num_clusters=args.num_clusters,
            num_countries=args.num_countries,
        )
    )

    _, val_loader = (
        create_cluster_country_dataloaders(
            train_csv=(
                "geo_dataset/train_cluster64.csv"
            ),
            val_csv=(
                "geo_dataset/val_cluster64.csv"
            ),
            image_dir=(
                "geo_dataset/train"
            ),
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

    all_cluster_log_probs = []
    all_country_log_probs = []

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

            cluster_log_probs = (
                F.log_softmax(
                    cluster_logits,
                    dim=1,
                )
            )

            country_log_probs = (
                F.log_softmax(
                    country_logits,
                    dim=1,
                )
            )

            all_cluster_log_probs.append(
                cluster_log_probs.cpu()
            )

            all_country_log_probs.append(
                country_log_probs.cpu()
            )

    cluster_log_probs = torch.cat(
        all_cluster_log_probs
    ).numpy()

    country_log_probs = torch.cat(
        all_country_log_probs
    ).numpy()

    centers_xyz = np.load(
        "geo_dataset/"
        "cluster64_centers_xyz.npy"
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    true_coordinates = torch.tensor(
        val_df[
            ["lat", "lng"]
        ].values,
        dtype=torch.float32,
    )

    alphas = [
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]

    results = []

    print()
    print(
        f"{'alpha':>7} "
        f"{'median':>10} "
        f"{'mean':>10} "
        f"{'<200':>9} "
        f"{'<750':>9} "
        f"{'>=750':>9}"
    )

    print("-" * 61)

    best_result = None

    for alpha in alphas:
        result = evaluate_alpha(
            cluster_log_probs=(
                cluster_log_probs
            ),
            country_log_probs=(
                country_log_probs
            ),
            cluster_country_probs=(
                cluster_country_probs
            ),
            centers_latlng=(
                centers_latlng
            ),
            true_coordinates=(
                true_coordinates
            ),
            alpha=alpha,
        )

        results.append(result)

        print(
            f"{alpha:7.2f} "
            f"{result['median_km']:10.2f} "
            f"{result['mean_km']:10.2f} "
            f"{result['within_200']:8.2f}% "
            f"{result['within_750']:8.2f}% "
            f"{result['catastrophic']:8.2f}%"
        )

        if (
            best_result is None
            or result["median_km"]
            < best_result["median_km"]
        ):
            best_result = result

    print()
    print("=== Best fusion ===")
    print(
        "Alpha:",
        best_result["alpha"],
    )
    print(
        "Median:",
        f"{best_result['median_km']:.2f} km",
    )
    print(
        "Mean:",
        f"{best_result['mean_km']:.2f} km",
    )
    print(
        "<200:",
        f"{best_result['within_200']:.2f}%",
    )
    print(
        "<750:",
        f"{best_result['within_750']:.2f}%",
    )
    print(
        ">=750:",
        f"{best_result['catastrophic']:.2f}%",
    )

    summary_rows = []

    for result in results:
        summary_rows.append(
            {
                "alpha": (
                    result["alpha"]
                ),
                "median_km": (
                    result["median_km"]
                ),
                "mean_km": (
                    result["mean_km"]
                ),
                "within_200_pct": (
                    result["within_200"]
                ),
                "within_750_pct": (
                    result["within_750"]
                ),
                "catastrophic_pct": (
                    result["catastrophic"]
                ),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        "geo_dataset/"
        "country_fusion_summary.csv",
        index=False,
    )

    best_predictions = (
        best_result[
            "predicted_clusters"
        ]
    )

    best_distances = (
        best_result[
            "distances"
        ]
    )

    output_df = val_df.copy()

    output_df[
        "fused_predicted_cluster"
    ] = best_predictions

    output_df[
        "fused_distance_km"
    ] = best_distances

    output_df.to_csv(
        "geo_dataset/"
        "country_fusion_best_predictions.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(
        "geo_dataset/"
        "country_fusion_summary.csv"
    )
    print(
        "geo_dataset/"
        "country_fusion_best_predictions.csv"
    )


if __name__ == "__main__":
    main()