import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def latlng_to_xyz(lat, lng):
    lat = np.radians(lat)
    lng = np.radians(lng)

    x = np.cos(lat) * np.cos(lng)
    y = np.cos(lat) * np.sin(lng)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=1)


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


def haversine_km(
    lat1,
    lng1,
    lat2,
    lng2,
    radius=6371.0088,
):
    lat1 = np.radians(lat1)
    lng1 = np.radians(lng1)
    lat2 = np.radians(lat2)
    lng2 = np.radians(lng2)

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlng / 2) ** 2
    )

    return (
        2
        * radius
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0, 1)
            )
        )
    )


def process_k(
    k,
    train_df,
    val_df,
    train_xyz,
    val_xyz,
    output_dir,
    save_files,
):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    train_clusters = kmeans.fit_predict(
        train_xyz
    )

    val_clusters = kmeans.predict(
        val_xyz
    )

    centers_xyz = kmeans.cluster_centers_

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    predicted_val_coordinates = centers_latlng[
        val_clusters
    ]

    distances = haversine_km(
        val_df["lat"].values,
        val_df["lng"].values,
        predicted_val_coordinates[:, 0],
        predicted_val_coordinates[:, 1],
    )

    cluster_sizes = np.bincount(
        train_clusters,
        minlength=k,
    )

    result = {
        "k": k,
        "oracle_median_km": np.median(distances),
        "oracle_mean_km": np.mean(distances),
        "within_200_pct": (
            distances < 200
        ).mean() * 100,
        "within_750_pct": (
            distances < 750
        ).mean() * 100,
        "min_cluster_size": int(
            cluster_sizes.min()
        ),
        "median_cluster_size": float(
            np.median(cluster_sizes)
        ),
        "max_cluster_size": int(
            cluster_sizes.max()
        ),
    }

    if save_files:
        train_output = train_df.copy()
        val_output = val_df.copy()

        train_output["geo_cluster"] = (
            train_clusters
        )

        val_output["geo_cluster"] = (
            val_clusters
        )

        train_path = (
            output_dir
            / f"train_cluster{k}.csv"
        )

        val_path = (
            output_dir
            / f"val_cluster{k}.csv"
        )

        centers_path = (
            output_dir
            / f"cluster{k}_centers_xyz.npy"
        )

        train_output.to_csv(
            train_path,
            index=False,
        )

        val_output.to_csv(
            val_path,
            index=False,
        )

        np.save(
            centers_path,
            centers_xyz,
        )

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        default=[
            24,
            32,
            36,
            48,
            64,
            80,
            96,
            128,
        ],
    )

    parser.add_argument(
        "--save-files",
        action="store_true",
        help=(
            "Save train/val cluster CSVs "
            "and cluster center files."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(
        "geo_dataset"
    )

    train_df = pd.read_csv(
        output_dir / "train_split.csv"
    )

    val_df = pd.read_csv(
        output_dir / "val_split.csv"
    )

    train_xyz = latlng_to_xyz(
        train_df["lat"].values,
        train_df["lng"].values,
    )

    val_xyz = latlng_to_xyz(
        val_df["lat"].values,
        val_df["lng"].values,
    )

    results = []

    print(
        f"{'K':>5} "
        f"{'median':>10} "
        f"{'mean':>10} "
        f"{'<200':>9} "
        f"{'<750':>9} "
        f"{'min':>7} "
        f"{'med_size':>10} "
        f"{'max':>7}"
    )

    print("-" * 77)

    for k in args.clusters:
        result = process_k(
            k=k,
            train_df=train_df,
            val_df=val_df,
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            output_dir=output_dir,
            save_files=args.save_files,
        )

        results.append(result)

        print(
            f"{result['k']:5d} "
            f"{result['oracle_median_km']:10.2f} "
            f"{result['oracle_mean_km']:10.2f} "
            f"{result['within_200_pct']:8.2f}% "
            f"{result['within_750_pct']:8.2f}% "
            f"{result['min_cluster_size']:7d} "
            f"{result['median_cluster_size']:10.1f} "
            f"{result['max_cluster_size']:7d}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_dir / "cluster_oracle_summary.csv",
        index=False,
    )

    print()
    print(
        "Saved summary to "
        "geo_dataset/cluster_oracle_summary.csv"
    )

    if args.save_files:
        print(
            "Also saved cluster CSVs and "
            "center files for every K."
        )


if __name__ == "__main__":
    main()