from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GeoDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image_path = self.image_dir / row["filename"]
        image = Image.open(image_path).convert("RGB")

        lat = row["lat"]
        lng = row["lng"]

        target = torch.tensor([lat, lng], dtype=torch.float32)

        image = self.transform(image)

        return image, target