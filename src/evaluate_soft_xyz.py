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

    hyp = np.sqrt(
        x**2 + y**2
    )

    lat = np.arctan2(
        z,
        hyp,
    )

    return np.stack(
        [
            np.degrees(lat),
            np.degrees(lng),
        ],
        axis=1,
    )


def normalize_xyz(xyz):
    norms = np.linalg.norm(
        xyz,
        axis=1,
        keepdims=True,
    )

    norms = np.clip(
        norms,
        1e-12,
        None,
    )

    return xyz / norms


def compute_metrics(
    predicted_xyz,
    true_coordinates,
):
    predicted_xyz = normalize_xyz(
        predicted_xyz
    )

    predicted_latlng = xyz_to_latlng(
        predicted_xyz
    )

    predicted_coordinates = torch.tensor(
        predicted_latlng,
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    ).numpy()

    return {
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
        "distances": distances,
        "predicted_latlng": predicted_latlng,
    }


def hard_argmax_xyz(
    logits,
    centers_xyz,
):
    predicted_clusters = np.argmax(
        logits,
        axis=1,
    )

    return centers_xyz[
        predicted_clusters
    ]


def soft_xyz(
    logits,
    centers_xyz,
    temperature,
):
    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    probabilities = F.softmax(
        logits_tensor / temperature,
        dim=1,
    ).numpy()

    predicted_xyz = (
        probabilities
        @ centers_xyz
    )

    return predicted_xyz


def topk_soft_xyz(
    logits,
    centers_xyz,
    temperature,
    top_k,
):
    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    scaled_logits = (
        logits_tensor
        / temperature
    )

    top_values, top_indices = torch.topk(
        scaled_logits,
        k=top_k,
        dim=1,
    )

    top_probabilities = F.softmax(
        top_values,
        dim=1,
    )

    top_indices_np = (
        top_indices.numpy()
    )

    top_probabilities_np = (
        top_probabilities.numpy()
    )

    selected_centers = centers_xyz[
        top_indices_np
    ]

    predicted_xyz = (
        selected_centers
        * top_probabilities_np[:, :, None]
    ).sum(
        axis=1
    )

    return predicted_xyz


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
            train_csv=(
                "geo_dataset/"
                "train_cluster64.csv"
            ),
            val_csv=(
                "geo_dataset/"
                "val_cluster64.csv"
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

    all_cluster_logits = []

    with torch.no_grad():
        for (
            images,
            _,
            _,
        ) in val_loader:
            images = images.to(device)

            (
                cluster_logits,
                _,
            ) = model(images)

            all_cluster_logits.append(
                cluster_logits.cpu()
            )

    cluster_logits = torch.cat(
        all_cluster_logits
    ).numpy()

    centers_xyz = np.load(
        "geo_dataset/"
        "cluster64_centers_xyz.npy"
    )

    val_df = pd.read_csv(
        "geo_dataset/"
        "val_cluster64.csv"
    )

    true_coordinates = torch.tensor(
        val_df[
            ["lat", "lng"]
        ].values,
        dtype=torch.float32,
    )

    results = []

    print()
    print(
        f"{'method':<18} "
        f"{'temp':>7} "
        f"{'median':>10} "
        f"{'mean':>10} "
        f"{'<200':>9} "
        f"{'<750':>9} "
        f"{'>=750':>9}"
    )

    print("-" * 78)

    hard_xyz = hard_argmax_xyz(
        logits=cluster_logits,
        centers_xyz=centers_xyz,
    )

    hard_metrics = compute_metrics(
        predicted_xyz=hard_xyz,
        true_coordinates=true_coordinates,
    )

    print(
        f"{'hard_argmax':<18} "
        f"{'-':>7} "
        f"{hard_metrics['median_km']:10.2f} "
        f"{hard_metrics['mean_km']:10.2f} "
        f"{hard_metrics['within_200']:8.2f}% "
        f"{hard_metrics['within_750']:8.2f}% "
        f"{hard_metrics['catastrophic']:8.2f}%"
    )

    results.append(
        {
            "method": "hard_argmax",
            "temperature": np.nan,
            "top_k": np.nan,
            "median_km": (
                hard_metrics["median_km"]
            ),
            "mean_km": (
                hard_metrics["mean_km"]
            ),
            "within_200_pct": (
                hard_metrics["within_200"]
            ),
            "within_750_pct": (
                hard_metrics["within_750"]
            ),
            "catastrophic_pct": (
                hard_metrics["catastrophic"]
            ),
        }
    )

    temperatures = [
        0.25,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]

    best = {
        "method": "hard_argmax",
        "temperature": None,
        "top_k": None,
        **hard_metrics,
    }

    for temperature in temperatures:
        predicted_xyz = soft_xyz(
            logits=cluster_logits,
            centers_xyz=centers_xyz,
            temperature=temperature,
        )

        metrics = compute_metrics(
            predicted_xyz=predicted_xyz,
            true_coordinates=true_coordinates,
        )

        print(
            f"{'full_soft':<18} "
            f"{temperature:7.2f} "
            f"{metrics['median_km']:10.2f} "
            f"{metrics['mean_km']:10.2f} "
            f"{metrics['within_200']:8.2f}% "
            f"{metrics['within_750']:8.2f}% "
            f"{metrics['catastrophic']:8.2f}%"
        )

        results.append(
            {
                "method": "full_soft",
                "temperature": temperature,
                "top_k": np.nan,
                "median_km": (
                    metrics["median_km"]
                ),
                "mean_km": (
                    metrics["mean_km"]
                ),
                "within_200_pct": (
                    metrics["within_200"]
                ),
                "within_750_pct": (
                    metrics["within_750"]
                ),
                "catastrophic_pct": (
                    metrics["catastrophic"]
                ),
            }
        )

        if (
            metrics["median_km"]
            < best["median_km"]
        ):
            best = {
                "method": "full_soft",
                "temperature": temperature,
                "top_k": None,
                **metrics,
            }

    top_k_values = [
        2,
        3,
        5,
    ]

    for top_k in top_k_values:
        for temperature in temperatures:
            predicted_xyz = topk_soft_xyz(
                logits=cluster_logits,
                centers_xyz=centers_xyz,
                temperature=temperature,
                top_k=top_k,
            )

            metrics = compute_metrics(
                predicted_xyz=predicted_xyz,
                true_coordinates=(
                    true_coordinates
                ),
            )

            method_name = (
                f"top{top_k}_soft"
            )

            print(
                f"{method_name:<18} "
                f"{temperature:7.2f} "
                f"{metrics['median_km']:10.2f} "
                f"{metrics['mean_km']:10.2f} "
                f"{metrics['within_200']:8.2f}% "
                f"{metrics['within_750']:8.2f}% "
                f"{metrics['catastrophic']:8.2f}%"
            )

            results.append(
                {
                    "method": method_name,
                    "temperature": (
                        temperature
                    ),
                    "top_k": top_k,
                    "median_km": (
                        metrics["median_km"]
                    ),
                    "mean_km": (
                        metrics["mean_km"]
                    ),
                    "within_200_pct": (
                        metrics["within_200"]
                    ),
                    "within_750_pct": (
                        metrics["within_750"]
                    ),
                    "catastrophic_pct": (
                        metrics["catastrophic"]
                    ),
                }
            )

            if (
                metrics["median_km"]
                < best["median_km"]
            ):
                best = {
                    "method": method_name,
                    "temperature": (
                        temperature
                    ),
                    "top_k": top_k,
                    **metrics,
                }

    print()
    print("=== Best decoding ===")
    print("Method:", best["method"])

    print(
        "Temperature:",
        best["temperature"],
    )

    print(
        "Top-k:",
        best["top_k"],
    )

    print(
        "Median:",
        f"{best['median_km']:.2f} km",
    )

    print(
        "Mean:",
        f"{best['mean_km']:.2f} km",
    )

    print(
        "<200:",
        f"{best['within_200']:.2f}%",
    )

    print(
        "<750:",
        f"{best['within_750']:.2f}%",
    )

    print(
        ">=750:",
        f"{best['catastrophic']:.2f}%",
    )

    summary_path = (
        "geo_dataset/"
        "soft_xyz_summary.csv"
    )

    pd.DataFrame(
        results
    ).sort_values(
        "median_km"
    ).to_csv(
        summary_path,
        index=False,
    )

    predictions_path = (
        "geo_dataset/"
        "soft_xyz_best_predictions.csv"
    )

    best_df = val_df.copy()

    best_df[
        "predicted_lat"
    ] = best[
        "predicted_latlng"
    ][:, 0]

    best_df[
        "predicted_lng"
    ] = best[
        "predicted_latlng"
    ][:, 1]

    best_df[
        "distance_km"
    ] = best[
        "distances"
    ]

    best_df.to_csv(
        predictions_path,
        index=False,
    )

    print()
    print("Saved:")
    print(summary_path)
    print(predictions_path)


if __name__ == "__main__":
    main()