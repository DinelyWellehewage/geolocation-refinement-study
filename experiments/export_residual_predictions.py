import argparse

import numpy as np
import torch

from src.datasets.dataloaders_cluster_country_residual_xyz import (
    create_cluster_country_residual_xyz_dataloaders,
)
from src.models.model_cluster_country_residual_xyz import (
    GeoClusterCountryResidualXYZModel,
)
from src.training.train_cluster_country_residual_xyz import (
    validate_cluster_country_residual_xyz,
)
from src.utils.metrics import haversine_distance_km


RESIDUAL_SCALE = 0.05


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


def compute_metrics(
    predicted_xyz,
    true_coordinates,
):
    predicted_coordinates = xyz_to_latlng(
        predicted_xyz
    )

    predicted_coordinates = torch.tensor(
        predicted_coordinates,
        dtype=torch.float32,
    )

    distances = haversine_distance_km(
        predicted_coordinates,
        true_coordinates,
    )

    return {
        "median": torch.median(
            distances
        ).item(),
        "mean": torch.mean(
            distances
        ).item(),
        "within_200": (
            (distances < 200)
            .float()
            .mean()
            .item()
            * 100
        ),
        "within_750": (
            (distances < 750)
            .float()
            .mean()
            .item()
            * 100
        ),
    }


def print_metrics(
    name,
    metrics,
):
    print()
    print(name)

    print(
        f"Median: {metrics['median']:.2f} km"
    )

    print(
        f"Mean: {metrics['mean']:.2f} km"
    )

    print(
        f"<200 km: {metrics['within_200']:.2f}%"
    )

    print(
        f"<750 km: {metrics['within_750']:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

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
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    from pathlib import Path
    output = Path(args.output_csv)
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Output already exists; choose a new filename.")


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    centers_path = (
        "geo_dataset/"
        "cluster128_centers_xyz.npy"
    )

    train_csv = (
        "geo_dataset/"
        "train_cluster128.csv"
    )

    val_csv = (
        "geo_dataset/"
        "val_cluster128.csv"
    )

    centers_xyz = np.load(
        centers_path
    ).astype(
        np.float32
    )

    _, val_loader = (
        create_cluster_country_residual_xyz_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            image_dir="geo_dataset/train",
            centers_path=centers_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=42,
            pretrained=True,
        )
    )

    model = GeoClusterCountryResidualXYZModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
        pretrained=False,
    ).to(
        device
    )

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
        )
    )

    val_metrics = (
        validate_cluster_country_residual_xyz(
            model=model,
            dataloader=val_loader,
            device=device,
        )
    )

    predicted_clusters = (
        val_metrics[
            "cluster_predictions"
        ]
        .numpy()
    )

    residual_predictions_scaled = (
        val_metrics[
            "residual_predictions"
        ]
        .numpy()
    )

    predicted_residuals = (
        RESIDUAL_SCALE
        * residual_predictions_scaled
    )

    cluster_only_xyz = (
        centers_xyz[
            predicted_clusters
        ]
    )

    refined_xyz = (
        cluster_only_xyz
        + predicted_residuals
    )

    refined_xyz = (
        refined_xyz
        / np.clip(
            np.linalg.norm(
                refined_xyz,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )
    )

    val_data = np.genfromtxt(
        val_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    true_coordinates = torch.tensor(
        np.column_stack(
            [
                val_data["lat"],
                val_data["lng"],
            ]
        ),
        dtype=torch.float32,
    )

    # Verify alignment before assigning image filenames.
    import hashlib
    import json
    import pandas as pd
    from torch.utils.data import SequentialSampler

    assert isinstance(val_loader.sampler, SequentialSampler)
    frame = pd.read_csv(val_csv)
    assert frame["filename"].is_unique
    targets = val_metrics["cluster_targets"].numpy()
    assert len(frame) == len(predicted_clusters)
    assert np.array_equal(targets, frame["geo_cluster"].to_numpy())

    center_coordinates = xyz_to_latlng(cluster_only_xyz)
    refined_coordinates = xyz_to_latlng(refined_xyz)

    center_errors = haversine_distance_km(
        torch.tensor(center_coordinates, dtype=torch.float32),
        true_coordinates,
    ).numpy()
    refined_errors = haversine_distance_km(
        torch.tensor(refined_coordinates, dtype=torch.float32),
        true_coordinates,
    ).numpy()

    assert np.isfinite(center_errors).all()
    assert np.isfinite(refined_errors).all()

    result = frame[
        ["filename", "country", "lat", "lng", "geo_cluster"]
    ].copy()
    result["predicted_cluster"] = predicted_clusters
    result["cluster_correct"] = predicted_clusters == targets
    result["center_lat"] = center_coordinates[:, 0]
    result["center_lng"] = center_coordinates[:, 1]
    result["refined_lat"] = refined_coordinates[:, 0]
    result["refined_lng"] = refined_coordinates[:, 1]
    result["center_error_km"] = center_errors
    result["refined_error_km"] = refined_errors
    result["delta_error_km"] = refined_errors - center_errors

    for index, axis in enumerate("xyz"):
        result[f"residual_{axis}"] = predicted_residuals[:, index]

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    metadata = {
        "checkpoint": args.checkpoint,
        "rows": len(result),
        "residual_scale": RESIDUAL_SCALE,
        "delta_definition": "refined_error_km minus center_error_km",
        "torch_version": torch.__version__,
        "device": str(device),
        "sha256": {
            str(path): sha256(path)
            for path in (
                args.checkpoint, val_csv, centers_path,
                "experiments/export_residual_predictions.py",
            )
        },
        "median_convention": "torch.median: lower middle for even counts",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        result.to_csv(stream, index=False)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)

    print("Exported:", output)
    print("Rows:", len(result))
    print("Negative delta means the residual improved that image.")

    cluster_metrics = compute_metrics(
        cluster_only_xyz,
        true_coordinates,
    )

    residual_metrics = compute_metrics(
        refined_xyz,
        true_coordinates,
    )

    print(
        "Cluster accuracy:",
        f"{val_metrics['cluster_accuracy'] * 100:.2f}%",
    )

    print(
        "Country accuracy:",
        f"{val_metrics['country_accuracy'] * 100:.2f}%",
    )

    print_metrics(
        "=== Cluster center only ===",
        cluster_metrics,
    )

    print_metrics(
        "=== Cluster center + residual ===",
        residual_metrics,
    )


if __name__ == "__main__":
    main()
    