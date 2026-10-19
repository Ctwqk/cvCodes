/*
 * Author: Taiwei Cui
 * Time: Feb 27 2024
 *
 * Shape features of a region (project 3, step 4).
 */
#ifndef REGIONFEATURES_H
#define REGIONFEATURES_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Number of values returned by featureBox.
constexpr int kShapeFeatureCount = 3;

// Features of the foreground of `map` (CV_8UC1, non-zero = region):
//   [0] fill ratio      pixel count / area of the oriented bounding box
//   [1] aspect ratio    short side / long side of the oriented box (rotation invariant, <= 1)
//   [2] sqrt(N) / (w+h) compactness: 0.5 for a square, 0.443 for a disc, small for thin shapes
// `ans` receives the oriented bounding box, aligned with the principal axes of the region.
// If `label` or `savePath` is given the feature vector is appended to that CSV file.
// Returns an empty vector (and a default `ans`) for an empty map.
std::vector<float> featureBox(const cv::Mat& map, cv::RotatedRect& ans, const std::string& savePath = "",
                              const std::string& label = "");

// Draw the oriented box, the axis of least moment and the last feature value (and `name`) onto dst.
int drawBox(const cv::RotatedRect& box, const std::vector<float>& features, cv::Mat& dst,
            const std::string& name = "", int thickness = 1);

#endif
