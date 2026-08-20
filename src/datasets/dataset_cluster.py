from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class GeoClusterDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform,
    ):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image_path = self.image_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")

        cluster = torch.tensor(
            int(row["geo_cluster"]),
            dtype=torch.long,
        )

        image = self.transform(image)

        return image, cluster