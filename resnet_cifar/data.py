"""CIFAR-10 loading, splitting and augmentation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CIFAR_MEAN, CIFAR_STD, TrainConfig


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def denormalize(batch: torch.Tensor) -> torch.Tensor:
    """Undo `Normalize` so tensors can be displayed as images."""
    mean = torch.tensor(CIFAR_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch * std + mean).clamp(0, 1)


def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, validation, test) loaders.

    The validation set is a reproducible random subset of the official
    training split; it uses the *evaluation* transform (no augmentation).
    """
    full_train = datasets.CIFAR10(cfg.data_dir, train=True, download=True,
                                  transform=train_transform())
    full_val = datasets.CIFAR10(cfg.data_dir, train=True, download=False,
                                transform=eval_transform())
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True,
                                transform=eval_transform())

    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(full_train), generator=gen).tolist()
    val_idx, train_idx = perm[: cfg.val_size], perm[cfg.val_size:]

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(full_train, train_idx), shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(Subset(full_val, val_idx), shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
