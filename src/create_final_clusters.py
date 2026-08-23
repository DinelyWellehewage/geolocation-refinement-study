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

    return np.column_stack([x, y, z])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_csv = Path("geo_dataset/train_labels.csv")
    output_dir = Path("geo_dataset/final")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(input_csv)

    print("Training samples:", len(df))

    xyz = latlng_to_xyz(
        df["lat"].to_numpy(),
        df["lng"].to_numpy(),
    )

    print(
        f"Fitting KMeans with "
        f"K={args.num_clusters}..."
    )

    kmeans = KMeans(
        n_clusters=args.num_clusters,
        random_state=args.seed,
        n_init=10,
    )

    cluster_ids = kmeans.fit_predict(xyz)

    df["geo_cluster"] = cluster_ids

    centers = kmeans.cluster_centers_

    # Project centers back onto unit sphere.
    centers = centers / np.linalg.norm(
        centers,
        axis=1,
        keepdims=True,
    )

    csv_path = (
        output_dir
        / f"train_cluster{args.num_clusters}_full.csv"
    )

    centers_path = (
        output_dir
        / f"cluster{args.num_clusters}_centers_full.npy"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    np.save(
        centers_path,
        centers,
    )

    counts = (
        df["geo_cluster"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=== Final clustering ===")
    print("Samples:", len(df))
    print("Clusters:", len(counts))
    print(
        "Cluster size min/median/max:",
        counts.min(),
        float(counts.median()),
        counts.max(),
    )

    print()
    print("Saved:")
    print(csv_path)
    print(centers_path)


if __name__ == "__main__":
    main()