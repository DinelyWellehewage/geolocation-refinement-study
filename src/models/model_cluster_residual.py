import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class GeoClusterResidualModel(nn.Module):
    def __init__(self, num_clusters=64):
        super().__init__()

        self.backbone = mobilenet_v3_large(
            weights=None,
        )

        in_features = self.backbone.classifier[-1].in_features

        # Remove original classifier output layer.
        self.backbone.classifier[-1] = nn.Identity()

        self.cluster_head = nn.Linear(
            in_features,
            num_clusters,
        )

        self.residual_head = nn.Sequential(
            nn.Linear(
                in_features,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(
                128,
                2,
            ),
        )

    def forward(self, x):
        features = self.backbone(x)

        cluster_logits = self.cluster_head(
            features
        )

        residual = self.residual_head(
            features
        )

        return cluster_logits, residual