// Visualising the first convolution layer of the trained network (project 5, task 2).
#ifndef FILTERVIZ_H
#define FILTERVIZ_H

#include <opencv2/core.hpp>
#include <vector>

#include "npy.h"

namespace p5 {

// `filters`: shape (N, h, w); `image`: shape (rows, cols). Every filter is applied with cv::filter2D
// (correlation, like a CNN layer; reflected border, same size output). Returns false if the shapes are wrong.
bool applyFilters(const NpyArray& filters, const NpyArray& image, std::vector<cv::Mat>& responses,
                  std::vector<cv::Mat>* filterMats = nullptr, cv::Mat* imageMat = nullptr);

// Contact sheet: the image, then for every filter the (enlarged) kernel above its response.
cv::Mat filterSheet(const cv::Mat& image, const std::vector<cv::Mat>& filters, const std::vector<cv::Mat>& responses,
                    int columns = 5, int cell = 112);

}  // namespace p5

#endif
