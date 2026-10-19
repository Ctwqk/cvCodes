"""Export the first-layer filters of the trained network and the first MNIST training image as .npy files.

The C++ program `filterViz` (project5/filterviz.cpp, replaces the old cnpy based `filter.cpp`) reads them,
applies the filters with cv::filter2D and writes the responses.

    python export_filters.py                        # writes filters.npy and image.npy next to this file
    python export_filters.py --plot results/filters # also plots the filters (and the responses, if --responses is given)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mnist_model import HERE, load_weights


def first_layer_filters(weights: str | Path) -> np.ndarray:
    """(10, 5, 5) float32 array: conv1 weights of the trained network."""
    net = load_weights(weights)
    return net.conv1.weight.detach().squeeze(1).numpy().astype(np.float32)


def first_training_image(root: str | Path) -> np.ndarray:
    """(28, 28) float32 array in [0, 1]: the first MNIST training image."""
    import torchvision

    trainset = torchvision.datasets.MNIST(root=str(root), train=True, download=True)
    return (np.asarray(trainset[0][0], dtype=np.float32) / 255.0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(HERE / "mnist_model_weights.pth"))
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--plot", default="", help="directory for filters.png")
    p.add_argument("--responses", default="", help="folder with ansImage<i>.npy written by filterViz (plotted next to the filters)")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    filters = first_layer_filters(args.weights)
    image = first_training_image(args.data)
    np.save(out / "filters.npy", filters)
    np.save(out / "image.npy", image)
    print(f"filters {filters.shape}, image {image.shape} written to {out}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(filters)):
            ax = fig.add_subplot(5, 4, 2 * i + 1, xticks=[], yticks=[])
            ax.imshow(filters[i])
            ax.set_title(f"filter {i + 1}")
            resp = Path(args.responses) / f"ansImage{i + 1}.npy" if args.responses else None
            if resp is not None and resp.exists():
                ax = fig.add_subplot(5, 4, 2 * i + 2, xticks=[], yticks=[])
                ax.imshow(np.load(resp))
                ax.set_title(f"result {i + 1}")
        fig.tight_layout()
        Path(args.plot).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(args.plot) / "filters.png", dpi=100)
        print(f"wrote {Path(args.plot) / 'filters.png'}")


if __name__ == "__main__":
    main()
