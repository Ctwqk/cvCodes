# cvCodes

Computer vision coursework repository with multiple C++ and Python projects covering image processing, feature extraction, recognition, calibration, and neural-network-based experiments.

## Covered Topics

- Basic image display and interaction
- Color analysis and K-means clustering
- Feature extraction and image matching
- Region segmentation and object recognition
- Camera calibration and pose / feature robustness experiments
- DNN embedding and ONNX-based classification experiments

## Project Layout

```text
project1/   # Intro image processing and display utilities
project2/   # Feature extraction and image matching
project3/   # Segmentation, thresholding, denoising, recognition
project4/   # Calibration / robust feature experiments
project5/   # DNN embeddings and learned-feature experiments
hw4/        # Additional color and clustering work
projectFinal/  # Final project assets / experiments
```

Most directories contain their own `CMakeLists.txt`, `Makefile`, and project-specific assets.

## Tech Stack

- C++
- OpenCV
- CMake / Make
- Python (for selected experiments and preprocessing)
- ONNX / DNN model assets in later projects

## Getting Started

Build and run each project independently from its own directory. For example:

```bash
cd project3
make
./main
```

If you prefer CMake:

```bash
cmake -S . -B build
cmake --build build
```

## Notes

- This repository is organized as a collection of assignments and experiments rather than a single unified application.
- Input images, intermediate data files, and small helper binaries are committed alongside the source for reproducibility.
