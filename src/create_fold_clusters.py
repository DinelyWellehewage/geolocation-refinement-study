import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def latlng_to_xyz(lat, lng):
    lat = np.radians(lat)
    lng = np.radians(lng)

    x = np.cos(lat) * np.cos(lng)
    y = np.cos(lat) * np.sin(lng)
    z = np.sin(lat)

    return np.column_stack(
        [x, y, z]
    )


def assign_nearest_cluster(
    xyz,
    centers_xyz,
):
    similarities = (
        xyz @ centers_xyz.T
    )

    return np.argmax(
        similarities,
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds-csv",
        type=str,
        default="folds.csv",
    )

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

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    k = args.num_clusters

    if args.output_dir is None:
        output_dir = (
            f"geo_dataset/cv_cluster{k}"
        )
    else:
        output_dir = args.output_dir

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.folds_csv
    )

    required = {
        "filename",
        "country",
        "lat",
        "lng",
        "fold",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    fold_ids = sorted(
        df["fold"].unique()
    )

    print(
        "Samples:",
        len(df),
    )
    print(
        "Folds:",
        fold_ids,
    )
    print(
        "Clusters:",
        k,
    )

    for fold in fold_ids:
        print()
        print(
            "========================================"
        )
        print(
            f"CREATING FOLD {fold}"
        )
        print(
            "========================================"
        )

        train_df = (
            df[df["fold"] != fold]
            .copy()
            .reset_index(drop=True)
        )

        val_df = (
            df[df["fold"] == fold]
            .copy()
            .reset_index(drop=True)
        )

        train_xyz = latlng_to_xyz(
            train_df["lat"].to_numpy(),
            train_df["lng"].to_numpy(),
        )

        val_xyz = latlng_to_xyz(
            val_df["lat"].to_numpy(),
            val_df["lng"].to_numpy(),
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=20,
        )

        train_clusters = (
            kmeans.fit_predict(
                train_xyz
            )
        )

        centers_xyz = (
            kmeans.cluster_centers_
        )

        centers_xyz = (
            centers_xyz
            / np.linalg.norm(
                centers_xyz,
                axis=1,
                keepdims=True,
            )
        )

        val_clusters = (
            assign_nearest_cluster(
                val_xyz,
                centers_xyz,
            )
        )

        train_df[
            "geo_cluster"
        ] = train_clusters

        val_df[
            "geo_cluster"
        ] = val_clusters

        train_path = os.path.join(
            output_dir,
            f"train_fold{fold}.csv",
        )

        val_path = os.path.join(
            output_dir,
            f"val_fold{fold}.csv",
        )

        centers_path = os.path.join(
            output_dir,
            f"centers_fold{fold}.npy",
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
            centers_xyz,
        )

        counts = (
            train_df[
                "geo_cluster"
            ]
            .value_counts()
        )

        print(
            "Train samples:",
            len(train_df),
        )

        print(
            "Validation samples:",
            len(val_df),
        )

        print(
            "Cluster size min/median/max:",
            int(counts.min()),
            f"{counts.median():.1f}",
            int(counts.max()),
        )

        print(
            "Saved:",
            train_path,
        )

        print(
            "Saved:",
            val_path,
        )

        print(
            "Saved:",
            centers_path,
        )

    print()
    print(
        "All fold-specific clusters created."
    )


if __name__ == "__main__":
    main()