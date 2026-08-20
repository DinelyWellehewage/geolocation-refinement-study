import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class GeoClusterCountryModel(nn.Module):
    def __init__(
        self,
        num_clusters=64,
        num_countries=12,
    ):
        super().__init__()

        self.backbone = mobilenet_v3_large(
            weights=None,
        )

        in_features = self.backbone.classifier[-1].in_features

        # Remove the original ImageNet classifier output.
        self.backbone.classifier[-1] = nn.Identity()

        self.cluster_head = nn.Linear(
            in_features,
            num_clusters,
        )

        self.country_head = nn.Linear(
            in_features,
            num_countries,
        )

    def forward(self, x):
        features = self.backbone(x)

        cluster_logits = self.cluster_head(
            features
        )

        country_logits = self.country_head(
            features
        )

        return cluster_logits, country_logits