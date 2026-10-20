# Reproducible results

These are outputs from running the programs, not mockups. The full pipeline was verified on **2026-10-19** in Ubuntu 24.04 with GCC 13, CMake 3.28, OpenCV 4.6, Python 3.12, and PyTorch 2.8.0 / torchvision 0.23.0 CPU wheels. Containers were limited to 4 GB RAM and four CPUs; C++ builds used at most four jobs. Local heavy work used the workstation's global lock and a 6 GB systemd scope with no swap.

## Verification

From the repository root:

```bash
bash scripts/docker.sh bash -c 'bash scripts/verify.sh build-quickstart && python3 -m pytest project5/tests -q && bash scripts/demo.sh build-quickstart/bin results/demo'
```

- The top-level build produces **26 executables**.
- CTest: **8 suites, 150 test cases, all passing**. This covers distances, CSV parsing/writing, filters, thresholding, morphology, regions, recognition, calibration, homography direction, K-means, IDX/NPY parsing and supplied-model inference.
- All **9 standalone configurations** in `scripts/verify.sh` build successfully, including both final-project variants. Every program's `--help` exits successfully without accessing a camera.
- Python: **12 tests passing**, including Greek-head training and comparison of PyTorch with both the supplied ONNX model and a fresh export (maximum permitted score difference: `1e-3`). The legacy ONNX exporter emits deprecation warnings; it is retained for compatibility with OpenCV 4.6/opset 10.
- [Full-pipeline CI runs](https://github.com/Ctwqk/cvCodes/actions/workflows/ci.yml?query=branch%3Aportfolio-upgrade). Each successful run's `headless-results` artifact contains the generated gallery, full demo log, MNIST confusion matrix and CTest log.

The independent review reproduced the original 145 C++ cases, nine Python tests, all nine standalone builds, all eight gallery JPEGs byte for byte, and the full MNIST confusion matrix from committed files in a fresh directory. It then demonstrated failures for quantization level counts, K-means accumulator overflow, palettes larger than 800 colors, ragged-row scaling, DNN crop distortion, and Greek train/test overlap. The corrected suite has the passing counts above. Project 2 was also configured independently with tests enabled; both its own and common suites passed.

The review retained all **48 deleted source/configuration paths**, using byte-identical archival copies or explicit duplicate mappings, and restored the original Project 1 submission notes. See the [source archive and hash manifest](../archive/original-coursework/README.md). Generated binaries and duplicated data remain removed. The documented recognition example now uses `ball1.jpg` because `ball.jpg` produces no retained region; the headless demo exercises that command. Corrected square padding changes the DNN result from 16/26 to 18/26, and the included DNN feature CSV has been regenerated.

## Measurements

| Experiment | Executed input / settings | Measured result |
| --- | --- | --- |
| Filters | Included `project1/cathedral.jpeg`, `imgDisplay --all` | All 15 filter outputs written |
| Colour quantisation | Cathedral, K=8, block=4, seed=1 | 19,200 colour samples; reconstruction MSE **217.1** |
| CBIR | All **1,106** included photographs | Six feature databases rebuilt; histogram and supplied ResNet18 top-5 retrieval montages saved |
| Shape recognition | 26 region rows, 5 classes; scaled distance, 1 neighbour | Leave-one-row-out **21/26 (80.8%)** |
| DNN region recognition | Same 26 rows; supplied region network, cosine distance, 1 neighbour | Leave-one-row-out **18/26 (69.2%)** |
| Synthetic calibration | 20 seeded 640×480 views, true `fx=fy=520` | Board found in **20/20**; fitted focal length **519.681**; RMS reprojection **0.1363 px** |
| AR sequence | 12 new synthetic views, estimated intrinsics | Board found in **12/12**; mean corner reprojection **0.140 px** |
| Harris corners | Included cathedral, threshold 160 | **83** selected corners |
| MNIST | All **10,000** included test digits; shipped ONNX model | **9,881 correct (98.81%)**, 119 errors; [full confusion matrix](mnist-confusion-matrix.txt) |
| Greek transfer | 27 training images, 13 evaluated test images after excluding one training duplicate; 15 epochs, seed 0 | **5/13 (38.5%)** test accuracy; small-data transfer is weak |
| Learned filters | Supplied 10×5×5 kernels and 28×28 digit | Ten filter-response arrays and a contact sheet written |
| Pattern matching | Cathedral registered and queried | **401** extracted SIFT features, **387** matched inliers |
| AR games | Both variants, 12 synthetic frames each | Headless field overlays produced; interactive play not assessed |

The Greek experiment is reproduced with `python3 project5/greek_transfer.py --epochs 15 --seed 0 --out results/greek`; it is separate from the C++ gallery script. Its [predictions](greek-predictions.png) and [epoch metrics](greek-metrics.json) are retained, including the errors. `greek_test/alpha/alpha_005.png` is byte-identical to `greek_train/alpha/alpha_004.png`; the script preserves both files but excludes the test copy by SHA-256 and records it in `excluded_test_samples`. The earlier **6/14 (42.9%)** included this correctly classified training image. Excluding it leaves **5/13 (38.5%)**, with an unchanged training history. Exact-byte filtering does not establish independence of handwriting sources or detect transformed duplicates.

The SIFT demo is an identity-view smoke test; the unit tests separately validate a non-identity transform with permuted correspondence indices. It is not a viewpoint-robustness benchmark.

## Reading the gallery

`gallery/` contains eight JPEG contact sheets, about **600 KB total**. `scripts/gallery.py` only resizes, arranges and captions the PNGs produced by the programs. Reproduce the full run into `results/demo`, then copy the small JPEGs from `results/demo/gallery` to `docs/gallery` if updating the showcase. The much larger raw output and generated CSVs stay ignored under `results/`.

- `filters.jpg`: original, sepia/vignette, gradient magnitude and kaleidoscope.
- `retrieval.jpg`: query at left; histogram and embedding retrieval underneath each other. Distance scales are method-specific.
- `segmentation.jpg`: input, threshold, retained component and oriented bounding box.
- `ar.jpg`: synthetic distorted board view and estimated-pose star overlay.
- `mnist.jpg`: first 32 test digits and first 32 mistakes, captions `truth > prediction`.
- `convolutions.jpg`: learned kernels and their image-correlation responses.
- `kmeans.jpg`: original and eight-colour approximation.
- `final.jpg`: SIFT localisation and 3D snake field on synthetic camera input.

## Limits and provenance

Recognition numbers describe a tiny coursework collection. Filename prefixes supply labels; photos with no retained region are skipped and reported in the log. Curate filenames before adding new training photos. Multiple objects from a single photo create multiple rows, so leaving out one row can retain related regions. These scores are not an independent test-set benchmark. `recognize --hold-out` excludes all database rows from the same input photo. Shape features outperform the supplied DNN representation on this particular run.

Calibration/AR images are **synthetic**, generated from the included checkerboard. A small reprojection error does not establish physical-camera accuracy or precise recovery of every distortion coefficient. Live camera controls, real lighting, full game sessions and other operating systems were not verified. The final project's planned Qt GUI is still absent.

The shipped MNIST model was evaluated, not retrained from scratch for this upgrade. Training and fresh ONNX export remain available in Python; a small Greek-letter head is trained in the test suite. Model SHA-256 values:

```text
project5/mnist_model.onnx
79b8ca32093d8879f90a376cfdab4d318eab01f0a73b27761baf38ae6c98eee6
project5/mnist_model_weights.pth
66c96f48cbdcac33818ccded74bdb2a27ab0e3f70cc424e050066fd1cc75fddc
project3/NetWorkThings/or2d-normmodel-007.onnx
ce3a5be24392ace5ae3676e95a64dc684947b31e0956c298ab5110a512f66e24
```

Original images, CSV embeddings, models and course-code attribution were retained. The owner should choose a repository license and confirm redistribution rights for these materials.
