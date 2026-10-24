"""Matplotlib figures used by the evaluation script and the README."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES


def plot_history(history: dict, path: str | Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")

    axes[1].plot(epochs, np.array(history["train_acc"]) * 100, label="train")
    axes[1].plot(epochs, np.array(history["val_acc"]) * 100, label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="%")

    axes[2].plot(epochs, history["lr"], color="tab:green")
    axes[2].set(title="Learning rate (one-cycle)", xlabel="epoch", ylabel="lr")

    for ax in axes[:2]:
        ax.legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: str | Path, names: Sequence[str] = CLASS_NAMES) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="predicted", ylabel="true", title="Normalised confusion matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(f1: np.ndarray, path: str | Path, names: Sequence[str] = CLASS_NAMES) -> None:
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(names)[order], f1[order] * 100, color="tab:blue")
    ax.set(xlabel="F1 score (%)", title="Per-class F1")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction_grid(images: np.ndarray, preds, labels, confs, path: str | Path,
                         names: Sequence[str] = CLASS_NAMES, cols: int = 8) -> None:
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    for ax in np.ravel(axes):
        ax.axis("off")
    for k in range(n):
        ax = np.ravel(axes)[k]
        ax.imshow(images[k])
        ok = preds[k] == labels[k]
        title = f"{names[preds[k]]} {confs[k]:.0%}"
        if not ok:
            title += f"\n(true: {names[labels[k]]})"
        ax.set_title(title, fontsize=8, color="green" if ok else "red")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_samples(images: np.ndarray, labels: np.ndarray, path: str | Path,
                       names: Sequence[str] = CLASS_NAMES, per_class: int = 8) -> None:
    fig, axes = plt.subplots(len(names), per_class, figsize=(per_class * 1.1, len(names) * 1.1))
    for c, name in enumerate(names):
        idx = np.flatnonzero(labels == c)[:per_class]
        for k in range(per_class):
            ax = axes[c, k]
            ax.axis("off")
            if k < len(idx):
                ax.imshow(images[idx[k]])
            if k == 0:
                ax.text(-0.15, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.14, wspace=0.05, hspace=0.05)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
