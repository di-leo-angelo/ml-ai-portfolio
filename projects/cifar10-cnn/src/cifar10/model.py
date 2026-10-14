"""Compact CNN for 32x32 color image classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, max pooling, and dropout."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return self.dropout(x)


class CifarCNN(nn.Module):
    """Four conv blocks (32/64/128/256 ch) + a small classifier head.

    Image path: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2, then flattened into a
    512-unit hidden layer and a 10-way softmax head (~1.7M parameters).
    """

    def __init__(self, num_classes: int = 10, fc_dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
