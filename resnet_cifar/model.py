"""Network definition.

The design follows the "pre-activation bottleneck" idea:

    BN -> ReLU -> 1x1 conv (reduce) -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv (expand)

The block output is *added* to a shortcut. If the spatial size or the number
of channels changes, the shortcut is a strided 1x1 projection of the
(already activated) input; otherwise it is the identity.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .config import TrainConfig


class PreActBottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, ratio: int = 4) -> None:
        super().__init__()
        inner = max(out_ch // ratio, 8)

        self.norm1 = nn.BatchNorm2d(in_ch)
        self.reduce = nn.Conv2d(in_ch, inner, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inner)
        self.spatial = nn.Conv2d(inner, inner, kernel_size=3, stride=stride,
                                 padding=1, bias=False)

        self.norm3 = nn.BatchNorm2d(inner)
        self.expand = nn.Conv2d(inner, out_ch, kernel_size=1, bias=False)

        self.act = nn.ReLU(inplace=True)

        needs_projection = stride != 1 or in_ch != out_ch
        self.project = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if needs_projection else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.act(self.norm1(x))
        shortcut = self.project(pre) if self.project is not None else x

        out = self.reduce(pre)
        out = self.spatial(self.act(self.norm2(out)))
        out = self.expand(self.act(self.norm3(out)))
        return out + shortcut


class CifarResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        stem_width: int = 32,
        stage_widths: Sequence[int] = (64, 128, 256),
        stage_depths: Sequence[int] = (2, 2, 2),
        ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(stage_widths) != len(stage_depths):
            raise ValueError("stage_widths and stage_depths must have the same length")

        self.stem = nn.Conv2d(in_channels, stem_width, 3, padding=1, bias=False)

        layers = []
        prev = stem_width
        for idx, (width, depth) in enumerate(zip(stage_widths, stage_depths)):
            for block_idx in range(depth):
                # only the first block of stages 2..n halves the resolution
                stride = 2 if (idx > 0 and block_idx == 0) else 1
                layers.append(PreActBottleneck(prev, width, stride, ratio))
                prev = width
        self.stages = nn.Sequential(*layers)

        self.final_norm = nn.BatchNorm2d(prev)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)       # works for any input size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(prev, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.act(self.final_norm(x))
        x = torch.flatten(self.pool(x), 1)
        return self.fc(self.drop(x))     # raw logits (softmax is folded into the loss)


def build_model(cfg: TrainConfig, num_classes: int = 10) -> CifarResNet:
    return CifarResNet(
        num_classes=num_classes,
        stem_width=cfg.stem_width,
        stage_widths=cfg.stage_widths,
        stage_depths=cfg.stage_depths,
        ratio=cfg.bottleneck_ratio,
        dropout=cfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
