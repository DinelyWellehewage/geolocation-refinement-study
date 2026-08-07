import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class GeoLocationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = mobilenet_v3_small(weights=None)

        in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features,
            2,
        )

    def forward(self, x):
        return self.model(x)