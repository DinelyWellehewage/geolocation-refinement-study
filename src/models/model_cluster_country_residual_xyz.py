import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


class GeoClusterCountryResidualXYZModel(nn.Module):
    def __init__(
        self,
        num_clusters=128,
        num_countries=12,
        pretrained=False,
    ):
        super().__init__()

        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None

        self.backbone = mobilenet_v3_large(
            weights=weights,
        )

        in_features = (
            self.backbone.classifier[-1]
            .in_features
        )

        # Remove original ImageNet classifier.
        self.backbone.classifier[-1] = nn.Identity()

        self.cluster_head = nn.Linear(
            in_features,
            num_clusters,
        )

        self.country_head = nn.Linear(
            in_features,
            num_countries,
        )

        # Predict local correction in XYZ space.
        self.residual_head = nn.Linear(
            in_features,
            3,
        )

    def forward(
        self,
        x,
    ):
        features = self.backbone(
            x
        )

        cluster_logits = self.cluster_head(
            features
        )

        country_logits = self.country_head(
            features
        )

        residual_xyz = self.residual_head(
            features
        )

        return (
            cluster_logits,
            country_logits,
            residual_xyz,
        )