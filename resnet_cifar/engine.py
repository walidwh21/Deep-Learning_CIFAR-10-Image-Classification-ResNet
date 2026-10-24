"""One-epoch train / evaluate routines."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import AverageMeter


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """Run one pass over `loader`. Trains when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()      # one-cycle schedule steps per batch

            n = targets.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update((logits.argmax(1) == targets).float().mean().item(), n)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    """Return softmax probabilities, predicted labels and true labels."""
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
        labels.append(targets)
    probs = torch.cat(probs)
    return {"probs": probs, "preds": probs.argmax(1), "labels": torch.cat(labels)}
