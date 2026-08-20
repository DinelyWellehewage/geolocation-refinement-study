import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class GeoClusterModel(nn.Module):
    def __init__(self, num_clusters=64):
        super().__init__()

        self.model = mobilenet_v3_large(
            weights=None,
        )

        in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features,
            num_clusters,
        )

    def forward(self, x):
        return self.model(x)