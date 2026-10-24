"""Command line entry point:  python -m resnet_cifar.train --epochs 30"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from .config import TrainConfig
from .data import make_loaders
from .engine import run_epoch
from .model import build_model, count_parameters
from .utils import ensure_dir, pick_device, set_seed


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train a bottleneck ResNet on CIFAR-10")
    p.add_argument("--data-dir", default=defaults.data_dir)
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--max-lr", type=float, default=defaults.max_lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing)
    p.add_argument("--stage-widths", type=int, nargs="+", default=list(defaults.stage_widths))
    p.add_argument("--stage-depths", type=int, nargs="+", default=list(defaults.stage_depths))
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--device", default=defaults.device)
    a = p.parse_args()
    return TrainConfig(
        data_dir=a.data_dir, output_dir=a.output_dir, epochs=a.epochs,
        batch_size=a.batch_size, max_lr=a.max_lr, weight_decay=a.weight_decay,
        label_smoothing=a.label_smoothing, stage_widths=tuple(a.stage_widths),
        stage_depths=tuple(a.stage_depths), num_workers=a.num_workers,
        seed=a.seed, device=a.device,
    )


def fit(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = ensure_dir(cfg.output_dir)
    cfg.save(out_dir / "config.json")

    train_loader, val_loader, _ = make_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"device={device}  parameters={count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.max_lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, epochs=cfg.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.25)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_path = 0.0, out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scheduler, cfg.grad_clip)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        for key, val in zip(history, (tr_loss, tr_acc, va_loss, va_acc,
                                      optimizer.param_groups[0]["lr"])):
            history[key].append(val)

        flag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                        "epoch": epoch, "val_acc": va_acc}, best_path)
            flag = "  * saved"
        print(f"epoch {epoch:3d}/{cfg.epochs}  train {tr_loss:.3f}/{tr_acc:.3%}  "
              f"val {va_loss:.3f}/{va_acc:.3%}  ({time.time() - t0:.0f}s){flag}")

    with open(out_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"best validation accuracy: {best_acc:.3%}  ->  {best_path}")
    return history


def main() -> None:
    fit(parse_args())


if __name__ == "__main__":
    main()
