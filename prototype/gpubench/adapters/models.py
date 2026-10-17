"""CIFAR-sized models (32x32 input)."""
from __future__ import annotations

from torch import nn
import torch.nn.functional as F


class _Basic(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class _ResNet(nn.Module):
    def __init__(self, widths: list[int], blocks: list[int], num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(_Basic(cin, w, 2 if (i > 0 and j == 0) else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def resnet20(num_classes: int = 10) -> nn.Module:
    """He et al. 2016 CIFAR ResNet-20, ~0.27M parameters."""
    return _ResNet([16, 32, 64], [3, 3, 3], num_classes)


def resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted to CIFAR (3x3 stem, no max-pool), ~11M parameters."""
    return _ResNet([64, 128, 256, 512], [2, 2, 2, 2], num_classes)


def tiny_cnn(num_classes: int = 10) -> nn.Module:
    """Small model for smoke tests on CPU."""
    return nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                         nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))


MODELS = {"resnet20": resnet20, "resnet18": resnet18, "tiny_cnn": tiny_cnn}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"model '{name}' không có; chọn một trong {sorted(MODELS)}")
    return MODELS[name](num_classes)
