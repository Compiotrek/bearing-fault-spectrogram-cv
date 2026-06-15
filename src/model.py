"""PyTorch model definitions for spectrogram classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _validate_num_classes(num_classes: int) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")


class SmallSpectrogramCNN(nn.Module):
    """Compact convolutional baseline for one-channel spectrograms."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        _validate_num_classes(num_classes)
        self.features = nn.Sequential(
            self._convolution_block(1, 16),
            self._convolution_block(16, 32),
            self._convolution_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    @staticmethod
    def _convolution_block(
        input_channels: int,
        output_channels: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(torch.flatten(pooled, start_dim=1))


class ResNet18SpectrogramClassifier(nn.Module):
    """ResNet-18 classifier accepting one- or three-channel spectrograms."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        _validate_num_classes(num_classes)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, num_classes)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.fc.parameters():
                parameter.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch_size, channels, H, W)")
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        elif inputs.shape[1] != 3:
            raise ValueError("inputs must have 1 or 3 channels")
        return self.backbone(inputs)


def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a supported spectrogram classifier."""
    if model_name == "small_cnn":
        return SmallSpectrogramCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return ResNet18SpectrogramClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(
        f"unsupported model_name {model_name!r}; expected 'small_cnn' or 'resnet18'"
    )
