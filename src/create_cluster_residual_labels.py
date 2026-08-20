import numpy as np
import pandas as pd


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


def add_residuals(csv_path, output_path, centers_latlng):
    df = pd.read_csv(csv_path)

    cluster_ids = df["geo_cluster"].values

    assigned_centers = centers_latlng[
        cluster_ids
    ]

    df["residual_lat"] = (
        df["lat"].values
        - assigned_centers[:, 0]
    )

    df["residual_lng"] = (
        df["lng"].values
        - assigned_centers[:, 1]
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {output_path}")
    print(
        "Median |lat residual|:",
        np.median(np.abs(df["residual_lat"])),
    )
    print(
        "Median |lng residual|:",
        np.median(np.abs(df["residual_lng"])),
    )


def main():
    centers_xyz = np.load(
        "geo_dataset/cluster64_centers_xyz.npy"
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    add_residuals(
        "geo_dataset/train_cluster64.csv",
        "geo_dataset/train_cluster64_residual.csv",
        centers_latlng,
    )

    add_residuals(
        "geo_dataset/val_cluster64.csv",
        "geo_dataset/val_cluster64_residual.csv",
        centers_latlng,
    )


if __name__ == "__main__":
    main()