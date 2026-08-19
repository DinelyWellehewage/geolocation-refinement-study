import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class CountryConditionedGeoModel(nn.Module):
    def __init__(self, num_countries=12):
        super().__init__()

        self.backbone = mobilenet_v3_small(weights=None)

        in_features = self.backbone.classifier[-1].in_features

        # Remove MobileNet's final output layer so the backbone
        # returns its learned feature representation.
        self.backbone.classifier[-1] = nn.Identity()

        # Predict one of the 12 countries.
        self.country_head = nn.Linear(
            in_features,
            num_countries,
        )

        # Coordinate head receives:
        # MobileNet features + 12 country probabilities.
        self.coordinate_head = nn.Sequential(
            nn.Linear(
                in_features + num_countries,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(
                256,
                2,
            ),
        )

    def forward(self, x):
        features = self.backbone(x)

        country_logits = self.country_head(features)

        country_probabilities = torch.softmax(
            country_logits,
            dim=1,
        )

        conditioned_features = torch.cat(
            [
                features,
                country_probabilities,
            ],
            dim=1,
        )

        coordinates = self.coordinate_head(
            conditioned_features
        )

        return coordinates, country_logits