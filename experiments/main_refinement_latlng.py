
"""Angular residual parameterization using the existing XYZ-space objective."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments import main_refinement_controlled as runner
from src.models.model_cluster_country_residual_xyz import (
    GeoClusterCountryResidualXYZModel as XYZModel,
)
from src.training.train_cluster_country_residual_xyz import (
    validate_cluster_country_residual_xyz as original_validate,
)

ANGLE_SCALE = 0.10
XYZ_SCALE = 0.05


class LatLngModel(XYZModel):
    def __init__(self, num_clusters=128, num_countries=12, pretrained=False):
        super().__init__(num_clusters, num_countries, pretrained)
        # Preserve the random stream used by the XYZ constructor.
        with torch.random.fork_rng(devices=[]):
            self.residual_head = nn.Linear(
                self.residual_head.in_features, 2
            )
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        # Inherited forward returns tanh-bounded angular outputs.


def decode_xyz(raw_angles, centers):
    """Preserve centre radius; zero angular output gives exact zero correction."""
    latitude = torch.atan2(
        centers[:, 2],
        torch.linalg.vector_norm(centers[:, :2], dim=1),
    )
    longitude = torch.atan2(centers[:, 1], centers[:, 0])
    new_latitude = latitude + ANGLE_SCALE * raw_angles[:, 0]
    new_longitude = longitude + ANGLE_SCALE * raw_angles[:, 1]

    def unit_xyz(lat, lng):
        return torch.stack([
            torch.cos(lat) * torch.cos(lng),
            torch.cos(lat) * torch.sin(lng),
            torch.sin(lat),
        ], dim=1)

    radius = torch.linalg.vector_norm(centers, dim=1, keepdim=True)
    displacement = radius * (
        unit_xyz(new_latitude, new_longitude)
        - unit_xyz(latitude, longitude)
    )
    return centers + displacement


def train_latlng(
    model, dataloader, optimizer, device,
    country_weight=0.2, residual_weight=1.0, label_smoothing=0.1,
):
    model.train()
    centers = torch.tensor(
        np.load("geo_dataset/cluster128_centers_xyz.npy"),
        dtype=torch.float32, device=device,
    )
    cluster_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    country_loss_fn = nn.CrossEntropyLoss()
    residual_loss_fn = nn.SmoothL1Loss()
    totals = np.zeros(4, dtype=np.float64)
    cluster_correct = country_correct = count = 0

    for images, clusters, countries, targets_xyz_scaled in dataloader:
        images = images.to(device)
        clusters = clusters.to(device)
        countries = countries.to(device)
        targets_xyz_scaled = targets_xyz_scaled.to(device)

        optimizer.zero_grad()
        cluster_logits, country_logits, raw_angles = model(images)

        assigned_centers = centers[clusters]
        predicted_xyz_scaled = (
            decode_xyz(raw_angles, assigned_centers) - assigned_centers
        ) / XYZ_SCALE

        cluster_loss = cluster_loss_fn(cluster_logits, clusters)
        country_loss = country_loss_fn(country_logits, countries)
        residual_loss = residual_loss_fn(
            predicted_xyz_scaled, targets_xyz_scaled
        )
        loss = (
            cluster_loss + country_weight * country_loss
            + residual_weight * residual_loss
        )
        loss.backward()
        optimizer.step()

        totals += [
            loss.item(), cluster_loss.item(),
            country_loss.item(), residual_loss.item(),
        ]
        cluster_correct += (
            cluster_logits.argmax(1) == clusters
        ).sum().item()
        country_correct += (
            country_logits.argmax(1) == countries
        ).sum().item()
        count += clusters.size(0)

    return {
        "total_loss": totals[0] / len(dataloader),
        "cluster_loss": totals[1] / len(dataloader),
        "country_loss": totals[2] / len(dataloader),
        "residual_loss": totals[3] / len(dataloader),
        "cluster_accuracy": cluster_correct / count,
        "country_accuracy": country_correct / count,
    }


def validate_latlng(model, dataloader, device):
    result = original_validate(model, dataloader, device)
    centers = torch.tensor(
        np.load("geo_dataset/cluster128_centers_xyz.npy"),
        dtype=torch.float32,
    )
    predicted_centers = centers[result["cluster_predictions"]]
    with torch.no_grad():
        result["residual_predictions"] = (
            decode_xyz(result["residual_predictions"], predicted_centers)
            - predicted_centers
        ) / XYZ_SCALE
    # Return equivalent scaled XYZ corrections for the existing evaluator.
    return result


def main():
    if "--export" in sys.argv:
        sys.argv.remove("--export")
        from experiments import export_residual_predictions as exporter
        exporter.GeoClusterCountryResidualXYZModel = LatLngModel
        exporter.validate_cluster_country_residual_xyz = validate_latlng
        exporter.main()
        return

    original_parse_args = runner.parse_args

    def parse_args():
        args = original_parse_args()
        assert args.residual_weight == 1.0
        assert args.checkpoint_name.startswith("latlng_")
        root = Path("checkpoints/cluster_country_residual_xyz")
        best = root / args.checkpoint_name
        paths = [
            best,
            best.with_name(best.stem + "_last.pth"),
            best.with_name(best.stem + "_epoch0.pth"),
            best.with_suffix(".metadata.json"),
            best.with_suffix(".latlng_protocol.json"),
        ]
        for path in paths:
            if path.exists():
                raise FileExistsError(path)
        root.mkdir(parents=True, exist_ok=True)

        protocol = {
            "angular_scale_radians": ANGLE_SCALE,
            "xyz_loss_scale": XYZ_SCALE,
            "residual_outputs": 2,
            "loss": "SmoothL1 on decoded XYZ displacement / 0.05",
            "training_center": "assigned training cluster",
            "inference_center": "predicted cluster",
            "center_radius": "preserved before final unit normalization",
            "bounds_choice": "0.10 covers all audited training angular offsets",
            "limitations": [
                "Angular and XYZ heads have different correction domains",
                "Angular head has two outputs rather than three",
            ],
            "adapter_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "controlled_runner_sha256": hashlib.sha256(
                Path("experiments/main_refinement_controlled.py").read_bytes()
            ).hexdigest(),
        }
        with paths[-1].open("x") as stream:
            json.dump(protocol, stream, indent=2)
        print(json.dumps(protocol, indent=2))
        return args

    runner.parse_args = parse_args
    runner.GeoClusterCountryResidualXYZModel = LatLngModel
    runner.train_one_epoch_cluster_country_residual_xyz = train_latlng
    runner.validate_cluster_country_residual_xyz = validate_latlng
    runner.main()


if __name__ == "__main__":
    main()
