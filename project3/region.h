// Region extraction (project 3, step 3): connected components of a binary
// image, minus the ones that touch the image border.
#ifndef REGION_H
#define REGION_H

#include <opencv2/core.hpp>
#include <vector>

constexpr int kMaxRegions = 6;          // regions are colour coded B, G, R, cyan, yellow, magenta
constexpr int kEdgeBand = 6;            // regions touching this many outer pixel rows/columns are dropped
constexpr int kDefaultMinArea = 1000;   // smaller regions are ignored

// 4-connected component labelling with union-find (union by rank, path compression).
//   binary  CV_8UC1, non-zero = foreground
//   labels  CV_32S output: 0 = background, 1..count = component id
// Returns the number of components; `areas[i]` is the pixel count of component i + 1.
int labelRegions(const cv::Mat& binary, cv::Mat& labels, std::vector<int>& areas);

// Find the (at most `num`, clamped to kMaxRegions) largest foreground regions that do not touch the
// image border band and have at least `minArea` pixels.
//   dst    CV_8UC3 colour-coded picture of the regions (0 / 255 values)
//   result one CV_8U mask (0 / 255) per region, largest first; ties are broken by the position of
//          the first pixel (top-left first)
// `src` is not modified (the original divided it by 255 in place), and dst may alias src.
std::vector<cv::Mat> regionize(const cv::Mat& src, cv::Mat& dst, int num = kMaxRegions,
                               int minArea = kDefaultMinArea);

#endif
