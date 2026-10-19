"""Curate actual program outputs into compact JPEG contact sheets, without synthesising results."""
from pathlib import Path
import sys
import cv2
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/demo")
destination = root / "gallery"
destination.mkdir(parents=True, exist_ok=True)


def sheet(name, panels, columns=2, width=400, height=300):
    rows = (len(panels) + columns - 1) // columns
    canvas = np.full((rows * (height + 32), columns * width, 3), 245, np.uint8)
    for i, (label, path) in enumerate(panels):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        scale = min(width / img.shape[1], height / img.shape[0])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        x, y = (i % columns) * width, (i // columns) * (height + 32)
        dx, dy = (width - img.shape[1]) // 2, (height - img.shape[0]) // 2
        canvas[y+dy:y+dy+img.shape[0], x+dx:x+dx+img.shape[1]] = img
        cv2.putText(canvas, label, (x+10, y+height+22), cv2.FONT_HERSHEY_SIMPLEX, .5, (30,30,30), 1, cv2.LINE_AA)
    target = destination / (name + ".jpg")
    if not cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise OSError(target)
    print(f"{target}: {target.stat().st_size:,} bytes")


sheet("filters", [(name, root/"filters"/(name+".png")) for name in ("original", "sepia", "magnitude", "kaleidoscope")])
sheet("kmeans", [("Original", Path("project1/cathedral.jpeg")), ("8 colors, seed 1", root/"kmeans/cathedral_k8.png")])
sheet("retrieval", [("Chromaticity histogram", root/"retrieval/match_histogram_pic.0164.png"), ("ResNet18 embeddings (supplied)", root/"retrieval/match_dnn_pic.0164.png")], columns=1, width=1000, height=210)
sheet("segmentation", [(label, root/"segmentation"/("img3P3_"+suffix+".png")) for label, suffix in [("Input", "1_input"), ("Threshold", "2_binary"), ("Connected components", "4_regions"), ("Oriented regions", "5_boxes")]])
sheet("ar", [("Synthetic camera view", root/"video/frame_0004.png"), ("Estimated pose + virtual star", root/"ar/vid_00004.png")])
sheet("mnist", [("First test digits: true > predicted", root/"mnist/samples.png"), ("First mistakes: true > predicted", root/"mnist/errors.png")], columns=1, width=700, height=350)
sheet("convolutions", [("Learned kernels and their filter responses", root/"convolutions/filter_responses.png")], columns=1, width=700, height=500)
sheet("final", [("SIFT pattern localisation", root/"tracking/tracking_00000.png"), ("3D snake field on synthetic camera input", root/"snake3d/snake_00004.png")])
