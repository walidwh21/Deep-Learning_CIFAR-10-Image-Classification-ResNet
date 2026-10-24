"""Analytic parameter count (pure Python, no torch needed).

Mirrors resnet_cifar/model.py so the figure quoted in the README can be checked
without installing a deep-learning framework.
"""
import argparse


def block_params(cin, cout, stride, ratio=4):
    inner = max(cout // ratio, 8)
    n = 2 * cin + cin * inner            # norm1 + 1x1 reduce
    n += 2 * inner + 9 * inner * inner   # norm2 + 3x3
    n += 2 * inner + inner * cout        # norm3 + 1x1 expand
    if stride != 1 or cin != cout:
        n += cin * cout                  # projection shortcut
    return n


def total(stem=32, widths=(64, 128, 256), depths=(2, 2, 2), classes=10, ratio=4):
    n = 3 * 9 * stem
    prev = stem
    for i, (w, d) in enumerate(zip(widths, depths)):
        for b in range(d):
            n += block_params(prev, w, 2 if (i > 0 and b == 0) else 1, ratio)
            prev = w
    n += 2 * prev                         # final BN
    n += prev * classes + classes         # classifier
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2])
    a = ap.parse_args()
    print(f"{total(widths=a.widths, depths=a.depths):,}")
