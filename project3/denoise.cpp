#include "denoise.h"

#include <iostream>

// Note on the port: the original cleanUp() divided the *input* image by 255 in place (corrupting
// the caller's data), leaked nothing but cost three small cv::Mat operations per pixel. This
// version counts neighbours with plain loops and leaves `src` alone; the result is identical.
int cleanUp(const cv::Mat& src, cv::Mat& dst, int kernelSize, int operationType, int kernelType) {
    if (kernelSize < 1 || kernelSize % 2 == 0) {
        std::cerr << "expecting odd kernel size" << std::endl;
        return -1;
    }
    if (src.empty() || src.type() != CV_8UC1 || (operationType != 0 && operationType != 1)) return -1;
    const int k = kernelSize, half = k / 2;
    const int threshold = static_cast<int>(static_cast<float>(k) * k / 4 / (1 + kernelType));

    // which window cells count: all of them, or every other one (checkerboard); never the centre
    std::vector<unsigned char> counts(static_cast<size_t>(k) * k, 1);
    if (!kernelType) {
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                if ((i % 2) ^ (j % 2)) counts[static_cast<size_t>(i) * k + j] = 0;
    }
    counts[static_cast<size_t>(half) * k + half] = 0;

    cv::Mat bin = src != 0;  // 0 / 255
    cv::Mat out = bin.clone();
    for (int y = half; y < src.rows - half; y++) {
        for (int x = half; x < src.cols - half; x++) {
            const int v = bin.at<uchar>(y, x) ? 1 : 0;
            if (v != operationType) continue;
            int differing = 0;
            for (int i = 0; i < k; i++) {
                const uchar* row = bin.ptr<uchar>(y - half + i);
                for (int j = 0; j < k; j++) {
                    if (counts[static_cast<size_t>(i) * k + j] && ((row[x - half + j] ? 1 : 0) != operationType)) differing++;
                }
            }
            if (differing > threshold) out.at<uchar>(y, x) = (1 - operationType) ? 255 : 0;
        }
    }
    dst = out;
    return 0;
}

int denoise(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat a, b;
    const cv::Mat input = src.clone();  // allow dst to alias src
    if (cleanUp(input, a, 3, 1, 0)) return -1;
    if (cleanUp(a, b, 3, 0, 1)) return -1;
    if (cleanUp(b, a, 5, 1, 0)) return -1;
    if (cleanUp(a, b, 5, 0, 1)) return -1;
    if (cleanUp(b, a, 3, 1, 0)) return -1;
    if (cleanUp(a, dst, 3, 0, 1)) return -1;
    return 0;
}
