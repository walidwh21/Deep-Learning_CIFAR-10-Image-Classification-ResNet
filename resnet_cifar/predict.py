"""Classify a single image file:  python -m resnet_cifar.predict --run-dir runs/default cat.png"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import CIFAR_MEAN, CIFAR_STD, CLASS_NAMES, TrainConfig
from .model import build_model
from .utils import pick_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    run = Path(args.run_dir)
    cfg = TrainConfig.load(run / "config.json")
    device = pick_device("auto")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    top = probs.topk(args.top_k)
    for p_, idx in zip(top.values, top.indices):
        print(f"{CLASS_NAMES[idx]:>11s}  {p_:.1%}")


if __name__ == "__main__":
    main()
