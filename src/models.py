"""Model definitions: ResNet-18, MobileNetV2, ViT-B/16, TinyCNN."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)


class TinyCNN(nn.Module):
    """Depthwise-separable CNN with ~27.3K parameters."""

    def __init__(self, num_classes: int = 38):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True))

        def block(cin, cmid, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cmid, 3, stride=stride, padding=1, groups=cin, bias=False),
                nn.BatchNorm2d(cmid), nn.ReLU(inplace=True),
                nn.Conv2d(cmid, cout, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.b1 = block(24, 24, 48, stride=1)
        self.b2 = block(48, 48, 72, stride=2)
        self.b3 = block(72, 72, 96, stride=2)
        self.b4 = block(96, 96, 88, stride=2)
        self.fc = nn.Linear(88, num_classes)

    def forward_features(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        feat = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return feat, self.fc(feat)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)[1]


def build_model(name: str, num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in ("resnet18", "resnet"):
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name in ("mobilenet_v2", "mobilenet"):
        m = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name in ("vit_b16", "vit"):
        m = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "tinycnn":
        m = TinyCNN(num_classes=num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return m


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if model_name in ("resnet18", "resnet"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name in ("mobilenet_v2", "mobilenet"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name in ("vit_b16", "vit"):
        for p in model.heads.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"freeze_backbone not supported for {model_name}")


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
