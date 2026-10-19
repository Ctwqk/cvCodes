// Image features for content-based image retrieval (CS 5330 project 2).
//
// All functions take a BGR 8-bit image by const reference (the original
// versions converted the caller's image to float in place) and return 0 on
// success, -1 when the image is too small or has the wrong type.
#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include <opencv2/core.hpp>
#include <vector>

constexpr int kHistBins = 32;                      // bins per axis of the chromaticity histogram
constexpr int kHistSize = kHistBins * kHistBins;   // 1024
constexpr int kTextureAngleBins = 25;              // 15 degree bins over [0, 360]
constexpr int kTextureSize = kTextureAngleBins * kHistBins;  // 800
constexpr int kGaborSize = 33;
constexpr int kFourierBlock = 8;                   // side of each spectrum block
constexpr int kFourierSize = 4 * kFourierBlock * kFourierBlock;  // 256
constexpr int kMultiHistRois = 5;
constexpr int kMultiHistRoiSize = 21;              // ROI side in pixels
// Largest value of the ResNet18 embeddings in the provided CSV; the Fourier
// block is scaled by it so both halves of the "custom" feature share a range.
constexpr float kDnnMaxValue = 11.843754f;

// The central 7x7 pixels (a 7x7 CV_8UC3 copy).
int central7x7(const cv::Mat& bgr, cv::Mat& patch);

// Any Mat as a row-major float vector.
std::vector<float> mat2vec(const cv::Mat& m);

// 2-D g/r chromaticity histogram (32x32 counts, index g * 32 + r), black pixels go to bin 0.
// Total mass = number of pixels.
int histogram(const cv::Mat& bgr, std::vector<float>& feature);

// Five spatial regions (centre + four 21x21 corners), one chromaticity histogram each,
// concatenated: 5 * 1024 values. (The original pooled the five regions into one
// histogram, which threw the spatial layout away.)
int multiHist(const cv::Mat& bgr, std::vector<float>& feature);

// Texture: 2-D histogram of Sobel gradient orientation (25 bins of 15 degrees) x
// normalised magnitude (32 bins), summed over the three colour channels. 800 values.
int texture(const cv::Mat& bgr, std::vector<float>& feature);

// Cell of the texture histogram for a gradient with orientation `angleDeg` in [0, 360] and
// magnitude `mag`, with magnitudes normalised from [lo, hi] to [0, 1]. Always in [0, 800).
int textureBinIndex(float angleDeg, float mag, float lo, float hi);

// 33-bin histogram of the greyscale image filtered with a Gabor kernel.
int gaborHist(const cv::Mat& bgr, std::vector<float>& feature);

// Magnitude of the DFT: four 8x8 blocks around the DC component (the lowest
// frequencies of the unshifted spectrum), log-scaled and normalised, times kDnnMaxValue.
int fourierHist(const cv::Mat& bgr, std::vector<float>& feature);

#endif
