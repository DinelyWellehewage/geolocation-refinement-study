import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.utils.metrics import haversine_distance_km
import torch


def latlng_to_xyz(lat, lng):
    lat = np.radians(lat)
    lng = np.radians(lng)

    x = np.cos(lat) * np.cos(lng)
    y = np.cos(lat) * np.sin(lng)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=1)


def xyz_to_latlng(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lng = np.arctan2(y, x)
    hyp = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, hyp)

    return np.degrees(lat), np.degrees(lng)


def main():
    train_df = pd.read_csv("geo_dataset/train_split.csv")
    val_df = pd.read_csv("geo_dataset/val_split.csv")

    train_xyz = latlng_to_xyz(
        train_df["lat"].values,
        train_df["lng"].values,
    )

    val_xyz = latlng_to_xyz(
        val_df["lat"].values,
        val_df["lng"].values,
    )

    cluster_counts = [24, 36, 48, 64, 96, 128]

    print(
        f"{'K':>4} "
        f"{'median_km':>12} "
        f"{'mean_km':>12} "
        f"{'<200km':>10} "
        f"{'<750km':>10} "
        f"{'min_size':>10} "
        f"{'median_size':>12} "
        f"{'max_size':>10}"
    )

    for k in cluster_counts:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        train_clusters = kmeans.fit_predict(train_xyz)

        centers_xyz = kmeans.cluster_centers_

        norms = np.linalg.norm(
            centers_xyz,
            axis=1,
            keepdims=True,
        )

        centers_xyz = centers_xyz / norms

        center_lat, center_lng = xyz_to_latlng(
            centers_xyz
        )

        centers_latlng = np.stack(
            [center_lat, center_lng],
            axis=1,
        )

        # Assign validation points to the nearest cluster
        # using centers fitted only on training data.
        val_clusters = kmeans.predict(val_xyz)

        predicted = centers_latlng[val_clusters]

        targets = val_df[
            ["lat", "lng"]
        ].values

        distances = haversine_distance_km(
            torch.tensor(
                predicted,
                dtype=torch.float32,
            ),
            torch.tensor(
                targets,
                dtype=torch.float32,
            ),
        ).numpy()

        cluster_sizes = np.bincount(
            train_clusters,
            minlength=k,
        )

        print(
            f"{k:4d} "
            f"{np.median(distances):12.2f} "
            f"{np.mean(distances):12.2f} "
            f"{(distances < 200).mean() * 100:9.2f}% "
            f"{(distances < 750).mean() * 100:9.2f}% "
            f"{cluster_sizes.min():10d} "
            f"{np.median(cluster_sizes):12.1f} "
            f"{cluster_sizes.max():10d}"
        )


if __name__ == "__main__":
    main()