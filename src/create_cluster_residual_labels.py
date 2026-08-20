import argparse
import os

import numpy as np
import pandas as pd


def latlng_to_xyz(lat, lng):
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)

    x = (
        np.cos(lat_rad)
        * np.cos(lng_rad)
    )

    y = (
        np.cos(lat_rad)
        * np.sin(lng_rad)
    )

    z = np.sin(lat_rad)

    return np.stack(
        [
            x,
            y,
            z,
        ],
        axis=1,
    )


def add_residual_targets(
    input_csv,
    output_csv,
    centers_xyz,
):
    df = pd.read_csv(
        input_csv
    )

    required_columns = {
        "lat",
        "lng",
        "geo_cluster",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    cluster_ids = (
        df["geo_cluster"]
        .to_numpy(
            dtype=np.int64
        )
    )

    if cluster_ids.min() < 0:
        raise ValueError(
            "Found negative cluster ID."
        )

    if cluster_ids.max() >= len(
        centers_xyz
    ):
        raise ValueError(
            "Cluster ID exceeds number "
            "of cluster centers."
        )

    true_xyz = latlng_to_xyz(
        df["lat"].to_numpy(),
        df["lng"].to_numpy(),
    )

    cluster_xyz = centers_xyz[
        cluster_ids
    ]

    residual_xyz = (
        true_xyz
        - cluster_xyz
    )

    residual_norm = np.linalg.norm(
        residual_xyz,
        axis=1,
    )

    df["residual_x"] = (
        residual_xyz[:, 0]
    )

    df["residual_y"] = (
        residual_xyz[:, 1]
    )

    df["residual_z"] = (
        residual_xyz[:, 2]
    )

    df["residual_norm"] = (
        residual_norm
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    return {
        "samples": len(df),
        "mean_norm": float(
            residual_norm.mean()
        ),
        "median_norm": float(
            np.median(
                residual_norm
            )
        ),
        "max_norm": float(
            residual_norm.max()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=64,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    k = args.num_clusters

    centers_path = (
        f"geo_dataset/"
        f"cluster{k}_centers_xyz.npy"
    )

    train_input = (
        f"geo_dataset/"
        f"train_cluster{k}.csv"
    )

    val_input = (
        f"geo_dataset/"
        f"val_cluster{k}.csv"
    )

    train_output = (
        f"geo_dataset/"
        f"train_cluster{k}_residual.csv"
    )

    val_output = (
        f"geo_dataset/"
        f"val_cluster{k}_residual.csv"
    )

    if not os.path.exists(
        centers_path
    ):
        raise FileNotFoundError(
            centers_path
        )

    centers_xyz = np.load(
        centers_path
    )

    print("Clusters:", k)
    print(
        "Centers shape:",
        centers_xyz.shape,
    )

    train_stats = add_residual_targets(
        input_csv=train_input,
        output_csv=train_output,
        centers_xyz=centers_xyz,
    )

    val_stats = add_residual_targets(
        input_csv=val_input,
        output_csv=val_output,
        centers_xyz=centers_xyz,
    )

    print()
    print("=== Train residuals ===")
    print(
        "Samples:",
        train_stats["samples"],
    )
    print(
        "Mean residual norm:",
        f"{train_stats['mean_norm']:.6f}",
    )
    print(
        "Median residual norm:",
        f"{train_stats['median_norm']:.6f}",
    )
    print(
        "Max residual norm:",
        f"{train_stats['max_norm']:.6f}",
    )

    print()
    print("=== Validation residuals ===")
    print(
        "Samples:",
        val_stats["samples"],
    )
    print(
        "Mean residual norm:",
        f"{val_stats['mean_norm']:.6f}",
    )
    print(
        "Median residual norm:",
        f"{val_stats['median_norm']:.6f}",
    )
    print(
        "Max residual norm:",
        f"{val_stats['max_norm']:.6f}",
    )

    print()
    print("Saved:")
    print(train_output)
    print(val_output)


if __name__ == "__main__":
    main()