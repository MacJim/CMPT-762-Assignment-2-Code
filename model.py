import torch
from torch import nn
import torchvision.models


class DenseNet (nn.Module):
    def __init__(self, n_classes: int):
        super(DenseNet, self).__init__()

        self.dummy_network = torchvision.models.densenet121()
        self.dummy_network.classifier = nn.Linear(1024, n_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.dummy_network(x)
        return x
