import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.model_cluster_country_residual_xyz import (
    GeoClusterCountryResidualXYZModel,
)


RESIDUAL_SCALE = 0.05

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

        self.image_paths = sorted(
            self.image_dir.glob(
                "*.jpg"
            )
        )

        self.transform = transform

    def __len__(self):
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index,
    ):
        image_path = self.image_paths[
            index
        ]

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image = self.transform(
            image
        )

        return (
            image,
            image_path.name,
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
        required=True,
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="geo_dataset/holdout_public",
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

    dataloader = DataLoader(
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

    model = GeoClusterCountryResidualXYZModel(
        num_clusters=args.num_clusters,
        num_countries=args.num_countries,
        pretrained=False,
    ).to(device)

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
    ).astype(
        np.float32
    )

    filenames = []
    all_predicted_xyz = []

    with torch.no_grad():
        for (
            images,
            batch_filenames,
        ) in dataloader:
            images = images.to(
                device
            )

            (
                cluster_logits,
                _,
                residual_scaled,
            ) = model(
                images
            )

            predicted_clusters = (
                cluster_logits
                .argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            residual_scaled = (
                residual_scaled
                .cpu()
                .numpy()
            )

            residual_xyz = (
                RESIDUAL_SCALE
                * residual_scaled
            )

            predicted_xyz = (
                centers_xyz[
                    predicted_clusters
                ]
                + residual_xyz
            )

            predicted_xyz = (
                predicted_xyz
                / np.clip(
                    np.linalg.norm(
                        predicted_xyz,
                        axis=1,
                        keepdims=True,
                    ),
                    1e-12,
                    None,
                )
            )

            all_predicted_xyz.append(
                predicted_xyz
            )

            filenames.extend(
                batch_filenames
            )

    predicted_xyz = np.concatenate(
        all_predicted_xyz,
        axis=0,
    )

    predicted_latlng = xyz_to_latlng(
        predicted_xyz
    )

    output_df = pd.DataFrame(
        {
            "filename": filenames,
            "pred_lat": predicted_latlng[:, 0],
            "pred_lng": predicted_latlng[:, 1],
        }
    )

    output_df.to_csv(
        args.output,
        index=False,
    )

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
    