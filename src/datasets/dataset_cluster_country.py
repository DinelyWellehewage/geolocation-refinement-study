from pathlib import Path

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


class GeoClusterCountryDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform,
    ):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        unknown_countries = (
            set(self.labels["country"].unique())
            - set(COUNTRY_TO_INDEX.keys())
        )

        if unknown_countries:
            raise ValueError(
                "Unknown countries found in dataset: "
                f"{sorted(unknown_countries)}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image_path = (
            self.image_dir
            / row["filename"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        cluster = torch.tensor(
            int(row["geo_cluster"]),
            dtype=torch.long,
        )

        country_name = row["country"]

        country = torch.tensor(
            COUNTRY_TO_INDEX[country_name],
            dtype=torch.long,
        )

        image = self.transform(image)

        return image, cluster, country