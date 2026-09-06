from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


COUNTRY_TO_IDX = {
    "Belarus": 0,
    "Finland": 1,
    "France": 2,
    "Germany": 3,
    "Iceland": 4,
    "Italy": 5,
    "Norway": 6,
    "Poland": 7,
    "Spain": 8,
    "Sweden": 9,
    "Turkey": 10,
    "United_Kingdom": 11,
}


# Covers the observed training residual range:
# maximum residual norm was approximately 0.0405.
RESIDUAL_SCALE = 0.05


def latlng_to_xyz(
    lat,
    lng,
):
    lat = np.radians(
        lat
    )

    lng = np.radians(
        lng
    )

    x = (
        np.cos(lat)
        * np.cos(lng)
    )

    y = (
        np.cos(lat)
        * np.sin(lng)
    )

    z = np.sin(
        lat
    )

    return np.array(
        [
            x,
            y,
            z,
        ],
        dtype=np.float32,
    )


class GeoClusterCountryResidualXYZDataset(
    Dataset
):
    def __init__(
        self,
        csv_file,
        image_dir,
        centers_path,
        transform=None,
    ):
        self.df = pd.read_csv(
            csv_file
        )

        self.image_dir = Path(
            image_dir
        )

        self.transform = transform

        self.centers_xyz = np.load(
            centers_path
        ).astype(
            np.float32
        )

    def __len__(
        self,
    ):
        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):
        row = self.df.iloc[
            index
        ]

        image_path = (
            self.image_dir
            / row["filename"]
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        cluster = int(
            row["geo_cluster"]
        )

        country = COUNTRY_TO_IDX[
            row["country"]
        ]

        true_xyz = latlng_to_xyz(
            row["lat"],
            row["lng"],
        )

        center_xyz = (
            self.centers_xyz[
                cluster
            ]
        )

        residual_xyz = (
            true_xyz
            - center_xyz
        )

        # Scale the residual so that the
        # regression target has a useful
        # numerical magnitude.
        residual_xyz_scaled = (
            residual_xyz
            / RESIDUAL_SCALE
        ).astype(
            np.float32
        )

        return (
            image,
            torch.tensor(
                cluster,
                dtype=torch.long,
            ),
            torch.tensor(
                country,
                dtype=torch.long,
            ),
            torch.tensor(
                residual_xyz_scaled,
                dtype=torch.float32,
            ),
        )
    