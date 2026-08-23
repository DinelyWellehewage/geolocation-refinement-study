import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.model_cluster_country import (
    GeoClusterCountryModel,
)


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


class HoldoutDataset(Dataset):
    def __init__(
        self,
        image_dir,
        transform,
    ):
        self.image_dir = Path(
            image_dir
        )

        self.files = sorted(
            self.image_dir.glob("*.jpg")
        )

        self.transform = transform

    def __len__(self):
        return len(
            self.files
        )

    def __getitem__(
        self,
        idx,
    ):
        path = self.files[idx]

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        image = self.transform(
            image
        )

        return (
            image,
            path.name,
        )


def xyz_to_latlng(
    xyz,
):
    xyz = (
        xyz
        / np.linalg.norm(
            xyz,
            axis=1,
            keepdims=True,
        )
    )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lng = np.arctan2(
        y,
        x,
    )

    hyp = np.sqrt(
        x**2 + y**2
    )

    lat = np.arctan2(
        z,
        hyp,
    )

    return np.stack(
        [
            np.degrees(lat),
            np.degrees(lng),
        ],
        axis=1,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--centers",
        type=str,
        default=(
            "geo_dataset/final/"
            "cluster128_centers_full.npy"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=(
            "geo_dataset/holdout_public"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-countries",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    dataset = HoldoutDataset(
        image_dir=args.image_dir,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        "Holdout images:",
        len(dataset),
    )

    model = GeoClusterCountryModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
        pretrained=False,
    ).to(
        device
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    centers_xyz = np.load(
        args.centers
    )

    centers_latlng = xyz_to_latlng(
        centers_xyz
    )

    filenames = []
    predicted_clusters = []

    with torch.no_grad():
        for (
            images,
            names,
        ) in loader:
            images = images.to(
                device
            )

            (
                cluster_logits,
                _,
            ) = model(
                images
            )

            clusters = (
                cluster_logits
                .argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            predicted_clusters.extend(
                clusters.tolist()
            )

            filenames.extend(
                list(names)
            )

    predicted_clusters = np.array(
        predicted_clusters
    )

    predicted_latlng = (
        centers_latlng[
            predicted_clusters
        ]
    )

    output_df = pd.DataFrame(
        {
            "filename": filenames,
            "pred_lat": (
                predicted_latlng[
                    :,
                    0,
                ]
            ),
            "pred_lng": (
                predicted_latlng[
                    :,
                    1,
                ]
            ),
        }
    )

    output_df.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        "Predictions:",
        len(output_df),
    )

    print(
        "Saved:",
        args.output,
    )

    print()
    print(
        output_df.head()
    )


if __name__ == "__main__":
    main()