#!/usr/bin/env bash
# Run from the repository root. All inputs are included; no camera/download required.
set -euo pipefail
bin=${1:-build/bin}
out=${2:-results/demo}
mkdir -p "$out"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_FOR_THREADS_NUM=2
"$bin/imgDisplay" project1/cathedral.jpeg --all --max-width 600 --out "$out/filters"
"$bin/colors" project1/cathedral.jpeg 8 --seed 1 --out "$out/kmeans" --headless
"$bin/establishCsv" all --csv-dir "$out/cbir" --dnn-csv project2/featureCsvs/ResNet18_olym.csv
"$bin/imgMatch" project2/images/pic.0164.jpg histogram --csv-dir "$out/cbir" --out "$out/retrieval" --headless
"$bin/imgMatch" project2/images/pic.0164.jpg dnn --out "$out/retrieval" --headless
"$bin/segment" project3/Proj03Examples/img3P3.png --out "$out/segmentation" --headless
"$bin/buildDb" --features shape --db "$out/shape.csv"
"$bin/evalDb" --db "$out/shape.csv" --metric scaled --k 1 --matrix
"$bin/recognize" project3/trainset/ball1.jpg --db "$out/shape.csv" --hold-out --out "$out/recognition" --headless
"$bin/buildDb" --features dnn --db "$out/dnn.csv"
"$bin/evalDb" --db "$out/dnn.csv" --features dnn --metric cosine --k 1 --matrix
"$bin/synthBoard" --out "$out/calibration" --frames 20 --seed 1
"$bin/calibrate" --images "$out/calibration" --out "$out/intrinsics.txt" --headless
"$bin/synthBoard" --out "$out/video" --mode video --frames 12
"$bin/CalPos" --source "$out/video/frame_%04d.png" --calib "$out/intrinsics.txt" --out "$out/ar" --max-frames 12 --headless
"$bin/RobustFeature" --source project4/cathedral.jpeg --out "$out/harris" --max-frames 1 --headless
"$bin/mnistEval" --out "$out/mnist"
"$bin/filterViz" --out "$out/convolutions"
"$bin/buildPatterns" --images project1 --out "$out/patterns.txt"
"$bin/matchPatterns" --patterns "$out/patterns.txt" --source project1/cathedral.jpeg --out "$out/tracking" --max-frames 1 --headless
"$bin/snakegame" --source "$out/video/frame_%04d.png" --calib "$out/intrinsics.txt" --width 8 --height 5 --offset-y -5 --out "$out/snake" --max-frames 12 --headless
"$bin/snakegame3d" --source "$out/video/frame_%04d.png" --calib "$out/intrinsics.txt" --width 8 --height 5 --depth 2 --offset-y -5 --out "$out/snake3d" --max-frames 12 --headless
python3 scripts/gallery.py "$out"
