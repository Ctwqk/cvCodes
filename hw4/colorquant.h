// Color quantisation helpers built on the K-means implementation.
#pragma once
#include <opencv2/core.hpp>
#include <random>
#include <vector>

// Jitter sampling: one random pixel from every BxB block of the image.
std::vector<cv::Vec3b> sampleColors(const cv::Mat& bgr, int blockSize, std::mt19937& rng);

// The color in `colors` closest to `pix` (SSD in BGR).
cv::Vec3b closestColor(const cv::Vec3b& pix, const std::vector<cv::Vec3b>& colors);

// Replace every pixel by its closest palette color.
cv::Mat quantize(const cv::Mat& bgr, const std::vector<cv::Vec3b>& palette);

// Share of the image's pixels (0..1) closest to each palette color.
std::vector<double> paletteShares(const cv::Mat& bgr, const std::vector<cv::Vec3b>& palette);

// Horizontal strip with one swatch per color; swatch width is proportional to
// `shares` when given (same length as palette), otherwise equal.
cv::Mat paletteStrip(const std::vector<cv::Vec3b>& palette, const std::vector<double>& shares,
                     int width = 800, int height = 80);

// "#RRGGBB"
std::string toHex(const cv::Vec3b& bgr);

// Mean squared error between two same-sized images, per channel value.
double meanSquaredError(const cv::Mat& a, const cv::Mat& b);
