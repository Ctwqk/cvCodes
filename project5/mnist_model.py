"""MNIST digit network for project 5: model definition, training and ONNX export.

    python mnist_model.py                      # train 5 epochs, write weights + ONNX in results/training
    python mnist_model.py --epochs 1 --out-dir results/mnist --data results/data

The ONNX file is what the C++ programs (mnistEval, mnistLive, ...) load with OpenCV DNN.
Inputs are normalised with mean 0.5 / std 0.5, exactly like `mnist_blob` in C++ (project5/mnist.cpp).
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MNIST_NORMALIZE = ((0.5,), (0.5,))


class Net(nn.Module):
    """Two conv layers + two fully connected layers; outputs log-probabilities for the ten digits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_weights(path: str | Path, device: str = "cpu") -> Net:
    """Load a state dict saved by `train` into an eval-mode Net."""
    net = Net()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.to(device).eval()
    return net


def read_idx(path: str | Path) -> "np.ndarray":
    """Read an MNIST IDX file (images -> (N, 28, 28) uint8, labels -> (N,) uint8) without torchvision."""
    import numpy as np

    raw = Path(path).read_bytes()
    magic = int.from_bytes(raw[:4], "big")
    dims = magic & 0xFF  # 3 for images, 1 for labels
    if magic >> 8 != 0x8 or dims not in (1, 3):
        raise ValueError(f"{path}: not an IDX ubyte file (magic {magic:#x})")
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(dims))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dims)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {int(np.prod(shape))} bytes of data, found {data.size}")
    return data.reshape(shape)


def load_mnist_test(root: str | Path, limit: int | None = None):
    """The MNIST test split from `root/MNIST/raw` as a normalised (N, 1, 28, 28) float tensor and labels.

    Reads the IDX files that are checked into the repository directly, so nothing is downloaded.
    """
    raw = Path(root) / "MNIST" / "raw"
    images = read_idx(raw / "t10k-images-idx3-ubyte")[:limit]
    labels = read_idx(raw / "t10k-labels-idx1-ubyte")[:limit]
    x = torch.from_numpy(images.astype("float32") / 255.0).unsqueeze(1)
    x = (x - MNIST_NORMALIZE[0][0]) / MNIST_NORMALIZE[1][0]
    return x, torch.from_numpy(labels.astype("int64"))


def mnist_loader(root: str | Path, batch_size: int, train: bool, download: bool = True):
    """DataLoader over the MNIST training or test split (the training split is downloaded on demand)."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*MNIST_NORMALIZE)])
    dataset = torchvision.datasets.MNIST(root=str(root), train=train, download=download, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)


@torch.no_grad()
def average_loss(net: nn.Module, loader, criterion, device: str, max_batches: int | None = None) -> float:
    total, count = 0.0, 0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        total += criterion(net(images), labels).item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


@torch.no_grad()
def accuracy(net: nn.Module, loader, device: str = "cpu") -> float:
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (net(images).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def export_onnx(net: Net, path: str | Path, example: torch.Tensor) -> None:
    net.eval()
    torch.onnx.export(
        net,
        example,
        str(path),
        export_params=True,
        opset_version=10,
        **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def train(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")  # the original opened a blocking window with plt.show()
    import matplotlib.pyplot as plt
    import torch.optim as optim

    torch.manual_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = mnist_loader(args.data, args.batch_size, train=True)
    test_loader = mnist_loader(args.data, args.batch_size, train=False)

    net = Net().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    seen, next_eval = 0, 0
    examples, train_losses, test_losses = [], [], []
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            net.train()
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            seen += images.size(0)
            if seen >= next_eval:  # the original only evaluated when `seen` was a multiple of 1000 (every 8000 examples)
                net.eval()
                examples.append(seen)
                train_losses.append(average_loss(net, train_loader, criterion, device, max_batches=40))
                test_losses.append(average_loss(net, test_loader, criterion, device))
                next_eval += args.eval_every
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    net.eval()
    acc = accuracy(net, test_loader, device)
    print(f"Accuracy of the network on the 10000 test images: {100 * acc:.2f} %")

    plt.figure(figsize=(6, 4))
    plt.plot(examples, train_losses, color="blue", label="train")
    plt.plot(examples, test_losses, color="red", label="test")
    plt.xlabel("training examples seen")
    plt.ylabel("negative log likelihood")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mnist_loss.png", dpi=120)

    torch.save(net.state_dict(), out_dir / "mnist_model_weights.pth")
    example = next(iter(train_loader))[0][:1].to(device)
    export_onnx(net, out_dir / "mnist_model.onnx", example)
    print(f"wrote {out_dir / 'mnist_model_weights.pth'}, {out_dir / 'mnist_model.onnx'} and mnist_loss.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--eval-every", type=int, default=6400, help="evaluate the loss curves every N training examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=str(HERE / "data"), help="MNIST root (downloaded on demand)")
    p.add_argument("--out-dir", default="results/training", help="where weights, ONNX file and loss plot are written")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())


if __name__ == "__main__":
    main()
