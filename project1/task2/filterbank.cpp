#include "filterbank.h"

#include <opencv2/imgproc.hpp>

#include "FaceDetect/faceDetect.h"
#include "filter.h"

namespace {

int applyOriginal(const cv::Mat& in, cv::Mat& out) {
    out = in.clone();
    return 0;
}

int applyGray(const cv::Mat& in, cv::Mat& out) {
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    return 0;
}

int applySobel(const cv::Mat& in, cv::Mat& out, int (*sobel)(const cv::Mat&, cv::Mat&)) {
    cv::Mat s;
    if (sobel(in, s)) return -1;
    cv::convertScaleAbs(s, out);
    return 0;
}
int applySobelX(const cv::Mat& in, cv::Mat& out) { return applySobel(in, out, sobelX3x3); }
int applySobelY(const cv::Mat& in, cv::Mat& out) { return applySobel(in, out, sobelY3x3); }

int applyMagnitude(const cv::Mat& in, cv::Mat& out) {
    cv::Mat sx, sy;
    if (sobelX3x3(in, sx) || sobelY3x3(in, sy)) return -1;
    return magnitude(sx, sy, out);
}

int applyBlurQuantize(const cv::Mat& in, cv::Mat& out) { return blurQuantize(in, out); }

int applyFaces(const cv::Mat& in, cv::Mat& out) {
    cv::Mat grey;
    cv::cvtColor(in, grey, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    if (detectFaces(grey, faces)) return -1;
    out = in.clone();
    drawBoxes(out, faces);
    return 0;
}

// Zoom in on the largest detected face (the frame is returned unchanged when there is none).
int applyZoomFace(const cv::Mat& in, cv::Mat& out) {
    cv::Mat grey;
    cv::cvtColor(in, grey, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    if (detectFaces(grey, faces)) return -1;
    out = in.clone();
    if (faces.empty()) return 0;
    cv::Rect largest = faces[0];
    for (const cv::Rect& f : faces) {
        if (f.width > largest.width) largest = f;
    }
    return centralize(out, largest);
}

int applyKaleidoscope(const cv::Mat& in, cv::Mat& out) {
    const int num = 5;
    const int side = std::min(in.rows, in.cols);
    const int size = side / num;
    if (size < 2) return -1;
    const cv::Rect square((in.cols - side) / 2, (in.rows - side) / 2, side, side);
    cv::Mat tile = in(square).clone();
    return kaleidoscope(tile, out, makeKaleidoscopeMask(size), num, size);
}

}  // namespace

const std::vector<FilterEntry>& filterBank() {
    static const std::vector<FilterEntry> bank = {
        {'o', "original", "original picture", applyOriginal},
        {'g', "gray", "OpenCV greyscale (cv::cvtColor)", applyGray},
        {'h', "mygray", "own greyscale: 255 - average of B, G, R", [](const cv::Mat& i, cv::Mat& o) { return myRGB2Gray(i, o); }},
        {'p', "sepia", "sepia tone with a vignette", [](const cv::Mat& i, cv::Mat& o) { return sepia(i, o); }},
        {'b', "blur", "5x5 Gaussian blur (separable)", [](const cv::Mat& i, cv::Mat& o) { return blur5x5_2(i, o); }},
        {'n', "blurnaive", "5x5 Gaussian blur (pixel-wise 2-D)", [](const cv::Mat& i, cv::Mat& o) { return blur5x5_1(i, o); }},
        {'x', "sobelx", "Sobel X (absolute value)", applySobelX},
        {'y', "sobely", "Sobel Y (absolute value)", applySobelY},
        {'m', "magnitude", "Sobel gradient magnitude", applyMagnitude},
        {'l', "quantize", "blur + quantise to 10 levels", applyBlurQuantize},
        {'f', "faces", "Haar-cascade face boxes", applyFaces},
        {'c', "zoomface", "zoom on the largest face", applyZoomFace},
        {'r', "redgreen", "red/green colour-blindness simulation", [](const cv::Mat& i, cv::Mat& o) { return confuseRedGreen(i, o); }},
        {'k', "kaleidoscope", "5x5 mirrored tiles", applyKaleidoscope},
        {'e', "localmax", "11x11 maximum filter", [](const cv::Mat& i, cv::Mat& o) { return localMax(i, o); }},
    };
    return bank;
}

const FilterEntry* findFilterByKey(int key) {
    for (const FilterEntry& e : filterBank()) {
        if (e.key == key) return &e;
    }
    return nullptr;
}

const FilterEntry* findFilterByName(const std::string& name) {
    for (const FilterEntry& e : filterBank()) {
        if (name == e.name) return &e;
    }
    return nullptr;
}
