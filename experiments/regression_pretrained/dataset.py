from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LAT_MEAN = 52.65384082447373
LAT_STD = 9.87951259488397

LNG_MEAN = 11.316609908569
LNG_STD = 15.245180654282903


class GeoDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image_path = self.image_dir / row["filename"]
        image = Image.open(image_path).convert("RGB")

        lat = row["lat"]
        lng = row["lng"]

        normalized_lat = (lat - LAT_MEAN) / LAT_STD
        normalized_lng = (lng - LNG_MEAN) / LNG_STD

        target = torch.tensor(
            [normalized_lat, normalized_lng],
            dtype=torch.float32,
        )

        image = self.transform(image)

        return image, target