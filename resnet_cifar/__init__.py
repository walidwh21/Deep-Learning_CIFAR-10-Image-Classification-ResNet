"""resnet_cifar - a compact pre-activation bottleneck ResNet for CIFAR-10 (PyTorch)."""

from .config import TrainConfig
from .model import PreActBottleneck, CifarResNet, build_model

__all__ = ["TrainConfig", "PreActBottleneck", "CifarResNet", "build_model"]
__version__ = "1.0.0"
