// The project-3 pipeline in one call: threshold -> clean up -> regions -> shape features.
#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "region.h"

struct SegmentOptions {
    int maxSide = 640;               // shrink (keeping the aspect ratio) so the longer side is at most this; 0 = keep
    int channel = -1;                // plane to threshold, see toBinary (-1 = automatic)
    int invert = -1;                 // -1 automatic, 0 bright objects on dark background, 1 dark objects on bright background
    int maxRegions = kMaxRegions;
    int minArea = -1;                // smallest region in pixels; -1 = 0.35 % of the resized image (1000 px at 480x600, the original working size)
    bool clean = true;               // run denoise()
};

struct RegionInfo {
    cv::Mat mask;                    // CV_8U 0 / 255, same size as SegmentResult::resized
    cv::RotatedRect box;
    std::vector<float> features;     // featureBox() values
    int area = 0;
};

struct SegmentResult {
    cv::Mat resized;                 // BGR input after resizing
    cv::Mat binary;                  // thresholded (objects white)
    cv::Mat cleaned;                 // after denoise (objects white)
    cv::Mat colored;                 // colour coded regions (CV_8UC3)
    bool inverted = false;           // true if the polarity was flipped (dark objects)
    std::vector<RegionInfo> regions; // largest first
};

// Returns 0, or -1 for an empty / non-BGR input.
int segmentImage(const cv::Mat& bgr, const SegmentOptions& options, SegmentResult& result);

// The input with every region's oriented box drawn on it (and `labels[i]` next to region i if given).
cv::Mat annotate(const SegmentResult& result, const std::vector<std::string>& labels = {});

#endif
