"""Central place for every tunable hyper-parameter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple

CLASS_NAMES: Tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Per-channel statistics of the CIFAR-10 training set.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class TrainConfig:
    # --- data -------------------------------------------------------------
    data_dir: str = "./data"
    val_size: int = 5_000          # held out from the 50k training images
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 42

    # --- model ------------------------------------------------------------
    stem_width: int = 32
    stage_widths: Tuple[int, ...] = (64, 128, 256)   # output channels per stage
    stage_depths: Tuple[int, ...] = (2, 2, 2)        # residual blocks per stage
    bottleneck_ratio: int = 4                         # out_channels // ratio = inner width
    dropout: float = 0.1

    # --- optimisation -----------------------------------------------------
    epochs: int = 30
    max_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    grad_clip: float = 5.0

    # --- bookkeeping ------------------------------------------------------
    output_dir: str = "./runs/default"
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    extra: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for key in ("stage_widths", "stage_depths"):
            raw[key] = tuple(raw[key])
        return cls(**raw)
