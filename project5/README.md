# Project 5 — PyTorch to OpenCV DNN

The supplied digit network has 1→10 and 10→20 convolution layers, pooling/dropout, and 320→50→10 fully connected layers. Both runtimes use `(pixel / 255 - 0.5) / 0.5`. The original weights, ONNX model, filter arrays, MNIST test data and small Greek-letter dataset remain included.

```bash
build/bin/mnistEval --out results/mnist
build/bin/filterViz --out results/convolutions
build/bin/mnistLive --source 0
python3 project5/predict_samples.py --out results/python-digits
python3 project5/greek_transfer.py --epochs 15 --seed 0 --out results/greek
python3 -m pytest project5/tests -q
```

Use the [Docker environment](../README.md#run-everything), or install CPU wheels in your own virtual environment:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r project5/requirements.txt
```

The versions follow the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/). The optional training command `python3 project5/mnist_model.py --epochs 5 --out-dir results/training` downloads the training split; it does not overwrite the supplied model by default. `export_filters.py` downloads the first training example when regenerating arrays. C++ evaluation and the full headless demo download no models or datasets.

![Actual MNIST predictions and errors](../docs/gallery/mnist.jpg)

`mnistLive` thresholds and centres the largest digit component before classification. It is a single-digit demonstration, not OCR for arbitrary scenes. Greek transfer learning freezes the feature layers and learns a 3-class head on 27 images. One of the 14 supplied test files is byte-identical to a training image, so evaluation excludes it and reports **13 test images**. With the default 15 epochs and seed 0, this run scored **5/13 (38.5%)**, replacing the earlier 6/14 (42.9%) that included the duplicate. All original images remain included; the excluded path is recorded in the metrics. The small dataset has high sampling uncertainty and illustrates limited transfer and overfitting, not a reliable Greek OCR system.

C++ tests cover IDX/NPY parsing, preprocessing and inference; Python tests verify weights, transforms, frozen layers, training, and OpenCV parity with both shipped and freshly exported ONNX models. Filter visualisation uses image correlation for interpretability, rather than claiming to reproduce the full normalised layer including bias and pooling.

ONNX export deliberately uses the legacy exporter and opset 10 for OpenCV 4.6 compatibility. PyTorch reports deprecation warnings; migration to the newer exporter needs another cross-runtime parity check.
