"""Run the trained MNIST network on the first test images and save a picture of the predictions.

    python predict_samples.py                       # prints the first ten predictions
    python predict_samples.py --out results/mnist   # also writes predictions.png (3x3 grid)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mnist_model import HERE, load_mnist_test, load_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(HERE / "mnist_model_weights.pth"))
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--out", default="")
    args = p.parse_args()

    net = load_weights(args.weights)
    images, labels = load_mnist_test(args.data, limit=10)
    with torch.no_grad():
        predicted = net(images).argmax(dim=1)
    for i in range(10):
        print(f"{i + 1}: predict: {int(predicted[i])}, actual: {int(labels[i])}")

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6, 6))
        for i in range(9):
            ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
            ax.imshow(images[i].squeeze().numpy() * 0.5 + 0.5, cmap="gray")
            ax.set_title(f"predict: {int(predicted[i])}, actual: {int(labels[i])}")
        fig.tight_layout()
        fig.savefig(out / "predictions.png", dpi=110)
        print(f"wrote {out / 'predictions.png'}")


if __name__ == "__main__":
    main()
