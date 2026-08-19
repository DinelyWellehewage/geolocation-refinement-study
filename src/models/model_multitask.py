import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MultiTaskGeoLocationModel(nn.Module):
    def __init__(self, num_countries=12):
        super().__init__()

        self.backbone = mobilenet_v3_small(weights=None)

        in_features = self.backbone.classifier[-1].in_features

        # Keep MobileNet's feature-producing classifier layers,
        # but remove its final classification layer.
        self.backbone.classifier[-1] = nn.Identity()

        # Coordinate regression head
        self.coordinate_head = nn.Linear(
            in_features,
            2,
        )

        # Auxiliary country classification head
        self.country_head = nn.Linear(
            in_features,
            num_countries,
        )

    def forward(self, x):
        features = self.backbone(x)

        coordinates = self.coordinate_head(features)
        country_logits = self.country_head(features)

        return coordinates, country_logits