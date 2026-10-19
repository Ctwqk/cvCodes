// Clean-up of binary images (project 3, step 2).
#ifndef DENOISE_H
#define DENOISE_H

#include <opencv2/core.hpp>

// One grow / shrink pass with a "majority" kernel.
//   src, dst        binary CV_8UC1 images (0 / non-zero in, 0 / 255 out); src is not modified
//   kernelSize      odd window size
//   operationType   0 = grow, 1 = shrink
//   kernelType      1 = every neighbour in the window counts (8-neighbourhood for size 3),
//                   0 = only every other cell (the diagonal neighbours for size 3)
// A pixel of the operated-on value flips when more than kernelSize^2 / 4 / (1 + kernelType)
// (integer division) of the counted neighbours have the other value. The (kernelSize / 2)-wide
// border is left unchanged. Returns 0, or -1 for an even kernel size / wrong image type.
int cleanUp(const cv::Mat& src, cv::Mat& dst, int kernelSize, int operationType, int kernelType);

// Chain of shrink/grow passes (3, 5, 3) that removes speckle and fills pin-holes.
int denoise(const cv::Mat& src, cv::Mat& dst);

#endif
