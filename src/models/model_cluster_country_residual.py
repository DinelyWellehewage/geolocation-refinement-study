import torch
import torch.nn as nn
from torchvision import models


class GeoClusterCountryResidualModel(nn.Module):
    def __init__(
        self,
        num_clusters=64,
        num_countries=12,
        residual_scale=0.06,
    ):
        super().__init__()

        self.residual_scale = residual_scale

        backbone = models.mobilenet_v3_large(
            weights=None,
        )

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        feature_dim = (
            backbone.classifier[0].in_features
        )

        self.shared = nn.Sequential(
            nn.Linear(
                feature_dim,
                960,
            ),
            nn.Hardswish(),
            nn.Dropout(
                p=0.2,
            ),
        )

        self.cluster_head = nn.Linear(
            960,
            num_clusters,
        )

        self.country_head = nn.Linear(
            960,
            num_countries,
        )

        self.residual_head = nn.Linear(
            960,
            3,
        )

    def forward(
        self,
        x,
    ):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(
            x,
            1,
        )

        shared_features = self.shared(x)

        cluster_logits = self.cluster_head(
            shared_features
        )

        country_logits = self.country_head(
            shared_features
        )

        raw_residual = self.residual_head(
            shared_features
        )

        residual_xyz = (
            self.residual_scale
            * torch.tanh(
                raw_residual
            )
        )

        return (
            cluster_logits,
            country_logits,
            residual_xyz,
        )