"""Draw the README diagrams (pipeline, residual block, full network).

    python scripts/make_diagrams.py            # writes docs/images/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "data": "#dbeafe", "aug": "#fef3c7", "model": "#dcfce7",
    "opt": "#fce7f3", "eval": "#ede9fe", "edge": "#334155",
    "conv": "#bfdbfe", "norm": "#fde68a", "act": "#fecaca", "add": "#bbf7d0",
}


def box(ax, x, y, w, h, text, color, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=PALETTE["edge"], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, p0, p1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12, lw=1.2,
                                 color=PALETTE["edge"], connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls))


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.4); ax.axis("off")
    steps = [
        ("CIFAR-10\n50k train / 10k test", "data"),
        ("Split\n45k train / 5k val", "data"),
        ("Augment\ncrop · flip · jitter\nrandom erasing", "aug"),
        ("Bottleneck\nResNet", "model"),
        ("SGD + one-cycle LR\nlabel smoothing", "opt"),
        ("Evaluate\nreport · confusion\nfigures", "eval"),
    ]
    w, gap, y = 1.7, 0.4, 1.0
    for i, (txt, c) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        box(ax, x, y, w, 1.4, txt, PALETTE[c], fs=9, weight="bold")
        if i:
            arrow(ax, (x - gap + 0.03, y + 0.7), (x - 0.03, y + 0.7))
    ax.text(6.5, 3.05, "Training & evaluation pipeline", ha="center", fontsize=13, weight="bold")
    ax.text(6.5, 0.35, "best-validation checkpoint is the one evaluated on the test set",
            ha="center", fontsize=9, style="italic", color="#475569")
    fig.savefig(OUT / "pipeline.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def block():
    fig, ax = plt.subplots(figsize=(8, 8.2))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8.6); ax.axis("off")
    cx, w, h = 3.0, 2.2, 0.55
    layers = [
        ("input x", "data"), ("BatchNorm + ReLU", "norm"), ("1×1 conv  (C → C/4)", "conv"),
        ("BatchNorm + ReLU", "norm"), ("3×3 conv  (stride s)", "conv"),
        ("BatchNorm + ReLU", "norm"), ("1×1 conv  (C/4 → C')", "conv"),
    ]
    ys = [7.6 - i * 1.0 for i in range(len(layers))]
    for (t, c), y in zip(layers, ys):
        box(ax, cx - w / 2 + 0.6, y, w, h, t, PALETTE[c], fs=9)
    xs = cx + 0.6
    for a, b in zip(ys, ys[1:]):
        arrow(ax, (xs, a), (xs, b + h))
    box(ax, xs - 0.5, 0.55, 1.0, 0.55, "add  ⊕", PALETTE["add"], fs=10, weight="bold")
    arrow(ax, (xs, ys[-1]), (xs, 1.1))
    ax.text(xs, 0.15, "output  (C' × H/s × W/s)", ha="center", fontsize=9)

    # shortcut path taken after the first BN+ReLU
    sx = 6.6
    box(ax, sx - 0.9, 4.65, 1.8, 0.7, "1×1 conv, stride s\n(only if shape changes)", PALETTE["conv"], fs=8)
    arrow(ax, (xs + w / 2, ys[1] + h / 2), (sx, 5.35), rad=-0.25)
    arrow(ax, (sx, 4.65), (xs + 0.5, 0.82), rad=-0.2)
    ax.text(sx, 5.7, "shortcut", ha="center", fontsize=9, style="italic")
    ax.text(7.35, 2.2, "no shape change:\nshortcut = raw input x\n(identity)", ha="center",
            fontsize=8, color="#475569")
    ax.text(4, 8.45, "Pre-activation bottleneck block", ha="center", fontsize=13, weight="bold")
    fig.savefig(OUT / "residual_block.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def network():
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")
    items = [
        ("Input\n3×32×32", "data", 1.2), ("Stem\n3×3 conv\n32×32×32", "conv", 1.3),
        ("Stage 1\n2 blocks\n64×32×32", "model", 1.5), ("Stage 2\n2 blocks (s=2)\n128×16×16", "model", 1.6),
        ("Stage 3\n2 blocks (s=2)\n256×8×8", "model", 1.6), ("BN·ReLU\nGlobal avg\npool → 256", "norm", 1.4),
        ("Dropout\nLinear\n→ 10 logits", "eval", 1.3),
    ]
    x, y = 0.2, 1.4
    centers = []
    for txt, c, w in items:
        box(ax, x, y, w, 1.7, txt, PALETTE[c], fs=9, weight="bold")
        centers.append((x, x + w)); x += w + 0.42
    for (a0, a1), (b0, b1) in zip(centers, centers[1:]):
        arrow(ax, (a1 + 0.02, y + 0.85), (b0 - 0.02, y + 0.85))
    ax.text(7, 4.2, "Network overview (default configuration)", ha="center", fontsize=13, weight="bold")
    ax.text(7, 0.7, "stride-2 projection blocks halve the resolution and double the channels  ·  "
            "global average pooling means no hard-coded feature-map size",
            ha="center", fontsize=9, style="italic", color="#475569")
    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    pipeline(); block(); network()
    print("diagrams written to", OUT)
