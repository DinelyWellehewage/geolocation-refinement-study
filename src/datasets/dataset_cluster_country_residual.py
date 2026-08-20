import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


COUNTRY_TO_INDEX = {
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


class GeoClusterCountryResidualDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
    ):
        self.data = pd.read_csv(
            csv_file
        )

        self.image_dir = image_dir
        self.transform = transform

        required_columns = {
            "filename",
            "country",
            "geo_cluster",
            "residual_x",
            "residual_y",
            "residual_z",
        }

        missing = (
            required_columns
            - set(self.data.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

    def __len__(self):
        return len(
            self.data
        )

    def __getitem__(
        self,
        index,
    ):
        row = self.data.iloc[
            index
        ]

        image_path = os.path.join(
            self.image_dir,
            row["filename"],
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

        cluster = torch.tensor(
            int(
                row["geo_cluster"]
            ),
            dtype=torch.long,
        )

        country_name = str(
            row["country"]
        )

        if country_name not in COUNTRY_TO_INDEX:
            raise KeyError(
                f"Unknown country: {country_name}"
            )

        country = torch.tensor(
            COUNTRY_TO_INDEX[
                country_name
            ],
            dtype=torch.long,
        )

        residual = torch.tensor(
            [
                row["residual_x"],
                row["residual_y"],
                row["residual_z"],
            ],
            dtype=torch.float32,
        )

        return (
            image,
            cluster,
            country,
            residual,
        )