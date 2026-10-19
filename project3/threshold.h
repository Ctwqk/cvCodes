// Thresholding (project 3, step 1): turn a colour image into a binary image.
#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/core.hpp>

// All threshold finders take an 8-bit single channel image, store the chosen
// threshold t in `pos` (pixels > t are "upper", pixels <= t are "lower") and
// return a score (see each function).

// Otsu's method: t maximises the between-class variance. Returns that variance
// (computed on the normalised histogram).
float otsuThreshold(const cv::Mat& gray, int& pos);

// 2-means / ISODATA: iterate t = (mean of lower + mean of upper) / 2 until it
// no longer moves. Returns t. A flat image gives t = its grey level.
float kmeanThreshold(const cv::Mat& gray, int& pos);

// t = mean grey level. Returns the mean.
float averageThreshold(const cv::Mat& gray, int& pos);

// dst (CV_8U) = upper where src > threshold, 0 elsewhere.
int myThreshold(const cv::Mat& src, cv::Mat& dst, int threshold, int upper);

// Colour image -> binary image (0 / 255). `idx` selects the plane that is
// thresholded: 0 blue, 1 green, 2 red, 3 OpenCV greyscale; -1 (default) picks
// the plane with the largest Otsu between-class variance (the greyscale
// variance counts double). Bright pixels become 255. Returns 0, or -1 for
// input that is not CV_8UC3.
int toBinary(const cv::Mat& src, cv::Mat& dst, int idx = -1);

#endif
