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


def main():
    train_df = pd.read_csv(
        "geo_dataset/train_split.csv"
    )
    val_df = pd.read_csv(
        "geo_dataset/val_split.csv"
    )

    train_xyz = latlng_to_xyz(
        train_df["lat"].values,
        train_df["lng"].values,
    )

    val_xyz = latlng_to_xyz(
        val_df["lat"].values,
        val_df["lng"].values,
    )

    # Fit clusters ONLY on the training split.
    # This avoids leaking validation coordinates.
    kmeans = KMeans(
        n_clusters=36,
        random_state=42,
        n_init=20,
    )

    train_df["geo_cluster"] = kmeans.fit_predict(
        train_xyz
    )

    # Assign validation samples using the centers
    # learned from the training split.
    val_df["geo_cluster"] = kmeans.predict(
        val_xyz
    )

    train_df.to_csv(
        "geo_dataset/train_cluster36.csv",
        index=False,
    )

    val_df.to_csv(
        "geo_dataset/val_cluster36.csv",
        index=False,
    )

    np.save(
        "geo_dataset/cluster36_centers_xyz.npy",
        kmeans.cluster_centers_,
    )

    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))

    print("\nTrain cluster counts:")
    print(
        train_df["geo_cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved:")
    print("geo_dataset/train_cluster36.csv")
    print("geo_dataset/val_cluster36.csv")
    print("geo_dataset/cluster36_centers_xyz.npy")


if __name__ == "__main__":
    main()