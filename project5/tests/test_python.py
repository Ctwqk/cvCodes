"""Tests for the PyTorch side of project 5.

Run from the repository root:   pytest project5/tests
Needs torch, torchvision, numpy (see requirements.txt); the ONNX cross-check also needs OpenCV (cv2) and is
skipped without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import export_filters  # noqa: E402
import greek_transfer  # noqa: E402
import mnist_model  # noqa: E402


@pytest.fixture(scope="module")
def net():
    return mnist_model.load_weights(HERE / "mnist_model_weights.pth")


@pytest.fixture(scope="module")
def test_digits():
    return mnist_model.load_mnist_test(HERE / "data", limit=1000)


def test_forward_shape_and_log_probabilities():
    model = mnist_model.Net().eval()
    out = model(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(4), atol=1e-5)  # log_softmax over the classes


def test_read_idx_matches_the_shipped_files():
    images = mnist_model.read_idx(HERE / "data/MNIST/raw/t10k-images-idx3-ubyte")
    labels = mnist_model.read_idx(HERE / "data/MNIST/raw/t10k-labels-idx1-ubyte")
    assert images.shape == (10000, 28, 28)
    assert labels.shape == (10000,)
    assert labels[0] == 7


def test_read_idx_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad"
    bad.write_bytes(b"\x00\x00\x09\x03" + b"\x00" * 16)  # wrong element type
    with pytest.raises(ValueError):
        mnist_model.read_idx(bad)
    short = tmp_path / "short"
    short.write_bytes(b"\x00\x00\x08\x01" + (5).to_bytes(4, "big") + b"\x01\x02")  # 5 labels announced, 2 present
    with pytest.raises(ValueError):
        mnist_model.read_idx(short)


def test_trained_weights_are_accurate(net, test_digits):
    x, y = test_digits
    with torch.no_grad():
        accuracy = (net(x).argmax(dim=1) == y).float().mean().item()
    assert accuracy >= 0.97


def test_onnx_model_agrees_with_the_pytorch_weights(net, test_digits, tmp_path):
    cv2 = pytest.importorskip("cv2")
    x, _ = test_digits
    exported = tmp_path / "exported.onnx"
    mnist_model.export_onnx(net, exported, x[:1])
    exported_net = cv2.dnn.readNet(str(exported))
    exported_net.setInput(x[:1].numpy())
    with torch.no_grad():
        assert np.abs(exported_net.forward() - net(x[:1]).numpy()).max() < 1e-3
    onnx = cv2.dnn.readNet(str(HERE / "mnist_model.onnx"))
    with torch.no_grad():
        expected = net(x[:50]).numpy()
    for i in range(50):
        onnx.setInput(x[i : i + 1].numpy())
        got = onnx.forward().reshape(-1)
        assert np.abs(got - expected[i]).max() < 1e-3
        assert got.argmax() == expected[i].argmax()


def test_export_filters_match_the_checked_in_arrays(tmp_path):
    filters = export_filters.first_layer_filters(HERE / "mnist_model_weights.pth")
    assert filters.shape == (10, 5, 5) and filters.dtype == np.float32
    assert np.allclose(filters, np.load(HERE / "filters.npy"))
    assert np.load(HERE / "image.npy").shape == (28, 28)


def test_greek_transform_produces_a_normalised_28x28_image():
    dataset, _ = greek_transfer.greek_loader(HERE / "greek_train", batch_size=1, shuffle=False)
    assert dataset.classes == ["alpha", "beta", "gamma"]
    assert len(dataset) == 27
    image, label = dataset[0]
    assert image.shape == (1, 28, 28)
    assert label == 0


def test_greek_net_only_trains_the_new_output_layer(net):
    model = greek_transfer.GreekNet(net)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"out.weight", "out.bias"}
    out = model(torch.zeros(2, 1, 28, 28))
    assert out.shape == (2, 3)


def test_greek_transfer_learning_runs_and_fits_the_training_set(tmp_path):
    import argparse

    args = argparse.Namespace(
        weights=str(HERE / "mnist_model_weights.pth"),
        train=str(HERE / "greek_train"),
        test=str(HERE / "greek_test"),
        epochs=12,
        seed=0,
        out=str(tmp_path),
    )
    result = greek_transfer.run(args)
    assert result["test_samples"] == 14
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["training"][-1]["train_accuracy"] > 0.6  # far above the 1/3 chance level
    assert (tmp_path / "predictions.png").exists()
    assert (tmp_path / "metrics.json").exists()
