"""Transfer learning: re-use the MNIST network to recognise the Greek letters alpha, beta and gamma.

The convolutional layers and fc1 of the trained MNIST network are frozen; only a new 50 -> 3 output layer is
trained on the 27 hand-written training letters (project5/greek_train). Test files identical to training images
are excluded from evaluation and reported explicitly.

    python greek_transfer.py                       # prints per-epoch accuracy and the test result
    python greek_transfer.py --out results/greek   # also writes predictions.png and metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from mnist_model import HERE, Net, load_weights

# Normalisation constants of the course-supplied Greek transform (the MNIST network itself was trained with 0.5 / 0.5).
GREEK_MEAN, GREEK_STD = 0.1307, 0.3081


class GreekTransform:
    """Greek letter photo (RGB, 133x133) -> inverted 28x28 grey image that looks like an MNIST digit."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.rgb_to_grayscale(x)
        x = TF.affine(x, 0, (0, 0), 36 / 128, 0)
        x = TF.center_crop(x, (28, 28))
        return TF.invert(x)


def greek_loader(path: str | Path, batch_size: int, shuffle: bool):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((GREEK_MEAN,), (GREEK_STD,)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(str(path), transform=transform)
    return dataset, torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class GreekNet(nn.Module):
    """MNIST network with frozen features and a new 3-way output layer."""

    def __init__(self, mnist: Net) -> None:
        super().__init__()
        for param in mnist.parameters():
            param.requires_grad = False
        self.features1 = nn.Sequential(mnist.conv1, mnist.pool1)
        self.features2 = nn.Sequential(mnist.conv2, mnist.dropout, mnist.pool2)
        self.fc1 = mnist.fc1
        self.out = nn.Linear(50, 3)  # created after freezing, so it is trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.features1(x))
        x = F.relu(self.features2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.out(x), dim=1)


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> tuple[float, list[tuple[torch.Tensor, int, int]]]:
    model.eval()
    correct = total = 0
    samples = []
    for images, labels in loader:
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        samples.extend((img, int(l), int(p)) for img, l, p in zip(images, labels, predicted))
    return correct / max(total, 1), samples


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    train_set, train_loader = greek_loader(Path(args.train), batch_size=5, shuffle=True)
    test_set, test_loader = greek_loader(Path(args.test), batch_size=1, shuffle=False)
    assert train_set.classes == test_set.classes, "train and test folders must have the same classes"

    training_hashes = {hashlib.sha256(Path(path).read_bytes()).digest() for path, _ in train_set.samples}
    held_out_indices, excluded_test_samples = [], []
    for index, (path, _) in enumerate(test_set.samples):
        if hashlib.sha256(Path(path).read_bytes()).digest() in training_hashes:
            excluded_test_samples.append(Path(path).relative_to(test_set.root).as_posix())
        else:
            held_out_indices.append(index)
    excluded_test_samples.sort()
    print(f"excluded {len(excluded_test_samples)} test images identical to training images")
    for path in excluded_test_samples:
        print(f"  {path}")
    if not held_out_indices:
        raise ValueError("no held-out test images remain after excluding training duplicates")
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_set, held_out_indices), batch_size=1, shuffle=False,
    )

    model = GreekNet(load_weights(args.weights))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    history = []
    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        loss = torch.tensor(0.0)
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        history.append({"epoch": epoch + 1, "last_batch_loss": float(loss.detach()), "train_accuracy": correct / total})
        print(f"epoch {epoch + 1:2d}  loss {float(loss.detach()):.4f}  training accuracy {correct / total:.3f}")

    test_acc, samples = evaluate(model, test_loader)
    classes = train_set.classes
    print(f"test accuracy {100 * test_acc:.1f} % on {len(samples)} letters")
    for img, truth, pred in samples:
        print(f"  actual {classes[truth]:5s} predicted {classes[pred]}{'' if truth == pred else '   <-- wrong'}")
    result = {"classes": classes, "test_accuracy": test_acc, "test_samples": len(samples),
              "excluded_test_samples": excluded_test_samples, "training": history}

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cols = 5
        rows = (len(samples) + cols - 1) // cols
        fig = plt.figure(figsize=(2 * cols, 2.2 * rows))
        for i, (img, truth, pred) in enumerate(samples):
            ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
            ax.imshow(img.squeeze().numpy() * GREEK_STD + GREEK_MEAN, cmap="gray")  # undo the normalisation
            ax.set_title(f"{classes[pred]}" + ("" if truth == pred else f" (is {classes[truth]})"),
                         color="green" if truth == pred else "red", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "predictions.png", dpi=110)
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {out / 'predictions.png'} and metrics.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(HERE / "mnist_model_weights.pth"))
    p.add_argument("--train", default=str(HERE / "greek_train"))
    p.add_argument("--test", default=str(HERE / "greek_test"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="directory for predictions.png and metrics.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
