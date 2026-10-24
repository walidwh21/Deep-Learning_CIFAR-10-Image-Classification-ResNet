"""Evaluate a checkpoint and write figures + metrics.

    python -m resnet_cifar.evaluate --run-dir runs/default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import CLASS_NAMES, TrainConfig
from .data import denormalize, make_loaders
from .engine import collect_predictions
from .model import build_model
from .plots import (plot_confusion, plot_history, plot_per_class,
                    plot_prediction_grid)
from .utils import ensure_dir, pick_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--grid-size", type=int, default=32)
    args = p.parse_args()

    run = Path(args.run_dir)
    cfg = TrainConfig.load(run / "config.json")
    if args.data_dir:
        cfg.data_dir = args.data_dir
    set_seed(cfg.seed)
    device = pick_device(args.device)
    fig_dir = ensure_dir(run / "figures")

    ckpt = torch.load(run / "best.pt", map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    _, _, test_loader = make_loaders(cfg)
    out = collect_predictions(model, test_loader, device)
    y_true, y_pred = out["labels"].numpy(), out["preds"].numpy()

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    print(report)
    print(f"test accuracy: {acc:.3%}")

    with open(run / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"test_accuracy": acc, "best_epoch": ckpt["epoch"],
                   "best_val_accuracy": ckpt["val_acc"],
                   "per_class_f1": dict(zip(CLASS_NAMES, map(float, f1)))}, fh, indent=2)
    (run / "classification_report.txt").write_text(report, encoding="utf-8")

    if (run / "history.json").exists():
        plot_history(json.loads((run / "history.json").read_text()), fig_dir / "training_curves.png")
    plot_confusion(cm, fig_dir / "confusion_matrix.png")
    plot_per_class(f1, fig_dir / "per_class_f1.png")

    # a reproducible random sample of test images with predictions
    rng = np.random.default_rng(cfg.seed)
    pick = rng.choice(len(y_true), size=args.grid_size, replace=False)
    test_ds = test_loader.dataset
    imgs = torch.stack([test_ds[i][0] for i in pick])
    imgs = denormalize(imgs).permute(0, 2, 3, 1).numpy()
    confs = out["probs"][pick].max(1).values.numpy()
    plot_prediction_grid(imgs, y_pred[pick], y_true[pick], confs, fig_dir / "predictions.png")
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
