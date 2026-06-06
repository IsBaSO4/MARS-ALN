"""Shared 3D ResNet-50 encoder for MARS.

All four input volumes (Breast-T0, Breast-T1, ALN-T0, ALN-T1) share the same
encoder weights to learn a unified tumour representation space. A projection
head maps the 2048-dim ResNet-50 features down to ``out_dim`` (256).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EncoderConfig:
    in_channels: int = 4          # 4 DCE pharmacokinetic channels
    out_dim: int = 256
    dropout: float = 0.3
    pretrained_path: str | None = None


class Bottleneck3D(nn.Module):
    """3D ResNet-50 bottleneck block (1x1 -> 3x3 -> 1x1, expansion 4)."""

    expansion = 4

    def __init__(self, in_ch: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D50(nn.Module):
    """3D ResNet-50 backbone (Bottleneck layers [3, 4, 6, 3]) -> (B, 2048)."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,   64,  3, stride=1)   # -> 256 ch
        self.layer2 = self._make_layer(256, 128,  4, stride=2)   # -> 512 ch
        self.layer3 = self._make_layer(512, 256,  6, stride=2)   # -> 1024 ch
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)   # -> 2048 ch

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.feature_dim = 512 * Bottleneck3D.expansion           # 2048

    @staticmethod
    def _make_layer(in_ch: int, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [Bottleneck3D(in_ch, planes, stride)]
        for _ in range(1, n_blocks):
            layers.append(Bottleneck3D(planes * Bottleneck3D.expansion, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return x


class SharedEncoder(nn.Module):
    """Shared 3D ResNet-50 encoder with projection head."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.backbone = ResNet3D50(in_channels=cfg.in_channels)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, cfg.out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.out_dim, cfg.out_dim),
        )
        if cfg.pretrained_path is not None:
            self._load_pretrained(cfg.pretrained_path)

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        backbone_state = {}
        for k, v in state.items():
            k_clean = k.replace("module.", "")
            if k_clean.startswith("backbone."):
                k_clean = k_clean[len("backbone."):]
            if k_clean in self.backbone.state_dict():
                backbone_state[k_clean] = v
        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one 3D volume (B, C, D, H, W) -> (B, out_dim)."""
        feat = self.backbone(x)
        return self.projection(feat)
