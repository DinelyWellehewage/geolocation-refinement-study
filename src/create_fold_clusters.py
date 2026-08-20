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

    return np.stack(
        [x, y, z],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    fold = args.fold
    k = args.num_clusters

    folds_df = pd.read_csv(
        "folds.csv"
    )

    train_df = folds_df[
        folds_df["fold"] != fold
    ].copy()

    val_df = folds_df[
        folds_df["fold"] == fold
    ].copy()

    train_xyz = latlng_to_xyz(
        train_df["lat"].values,
        train_df["lng"].values,
    )

    val_xyz = latlng_to_xyz(
        val_df["lat"].values,
        val_df["lng"].values,
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=args.seed,
        n_init=20,
    )

    train_df["geo_cluster"] = kmeans.fit_predict(
        train_xyz
    )

    val_df["geo_cluster"] = kmeans.predict(
        val_xyz
    )

    output_dir = Path(
        f"geo_dataset/cv_cluster{k}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        output_dir
        / f"train_fold{fold}.csv"
    )

    val_path = (
        output_dir
        / f"val_fold{fold}.csv"
    )

    centers_path = (
        output_dir
        / f"centers_fold{fold}.npy"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    np.save(
        centers_path,
        kmeans.cluster_centers_,
    )

    cluster_sizes = (
        train_df["geo_cluster"]
        .value_counts()
        .sort_index()
    )

    print("Fold:", fold)
    print("Clusters:", k)
    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))
    print("Train clusters:", train_df["geo_cluster"].nunique())
    print("Val clusters:", val_df["geo_cluster"].nunique())
    print(
        "Cluster range:",
        train_df["geo_cluster"].min(),
        train_df["geo_cluster"].max(),
    )

    print()
    print("Cluster size summary:")
    print("Min:", cluster_sizes.min())
    print("Median:", cluster_sizes.median())
    print("Max:", cluster_sizes.max())

    print()
    print("Saved:")
    print(train_path)
    print(val_path)
    print(centers_path)


if __name__ == "__main__":
    main()