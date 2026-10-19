// Reader for the IDX files MNIST is distributed in (big-endian header, then raw bytes).
#ifndef IDX_H
#define IDX_H

#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace p5 {

// idx3-ubyte: `count` images (rows x cols, CV_8UC1). `limit` > 0 reads only the first `limit` images.
bool readIdxImages(const std::string& path, std::vector<cv::Mat>& images, std::string* err = nullptr, size_t limit = 0);
// idx1-ubyte: one label per byte.
bool readIdxLabels(const std::string& path, std::vector<uint8_t>& labels, std::string* err = nullptr, size_t limit = 0);

}  // namespace p5

#endif
