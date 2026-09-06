import torch.nn as nn

from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


class GeoRegressionPretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT,
        )

        in_features = (
            self.model.classifier[-1].in_features
        )

        self.model.classifier[-1] = nn.Linear(
            in_features,
            2,
        )

    def forward(self, x):
        return self.model(x)