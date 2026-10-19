#include "filterviz.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace p5 {

bool applyFilters(const NpyArray& filters, const NpyArray& image, std::vector<cv::Mat>& responses,
                  std::vector<cv::Mat>* filterMats, cv::Mat* imageMat) {
    if (filters.shape.size() != 3 || image.shape.size() != 2) return false;
    const int n = static_cast<int>(filters.shape[0]), h = static_cast<int>(filters.shape[1]), w = static_cast<int>(filters.shape[2]);
    const int rows = static_cast<int>(image.shape[0]), cols = static_cast<int>(image.shape[1]);
    if (n < 1 || h < 1 || w < 1 || rows < 1 || cols < 1 || filters.data.size() != static_cast<size_t>(n) * h * w ||
        image.data.size() != static_cast<size_t>(rows) * cols) {
        return false;
    }
    const cv::Mat img(rows, cols, CV_32F, const_cast<float*>(image.data.data()));
    responses.clear();
    if (filterMats) filterMats->clear();
    for (int i = 0; i < n; i++) {
        const cv::Mat kernel(h, w, CV_32F, const_cast<float*>(filters.data.data() + static_cast<size_t>(i) * h * w));
        cv::Mat out;
        cv::filter2D(img, out, -1, kernel);
        responses.push_back(out);
        if (filterMats) filterMats->push_back(kernel.clone());
    }
    if (imageMat) *imageMat = img.clone();
    return true;
}

namespace {
cv::Mat toTile(const cv::Mat& m, int cell) {
    cv::Mat u8, big, bgr;
    cv::normalize(m, u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::resize(u8, big, cv::Size(cell, cell), 0, 0, cv::INTER_NEAREST);
    cv::applyColorMap(big, bgr, cv::COLORMAP_VIRIDIS);
    return bgr;
}
}  // namespace

cv::Mat filterSheet(const cv::Mat& image, const std::vector<cv::Mat>& filters, const std::vector<cv::Mat>& responses,
                    int columns, int cell) {
    const size_t n = std::min(filters.size(), responses.size());
    const int cap = 18;
    const int perRow = columns;
    const int rowsOfFilters = static_cast<int>((n + perRow - 1) / perRow);
    cv::Mat sheet(cell + cap + rowsOfFilters * (2 * cell + 2 * cap), perRow * cell, CV_8UC3, cv::Scalar(255, 255, 255));
    toTile(image, cell).copyTo(sheet(cv::Rect(0, 0, cell, cell)));
    cv::putText(sheet, "input", cv::Point(4, cell + 13), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    for (size_t i = 0; i < n; i++) {
        const int r = static_cast<int>(i) / perRow, c = static_cast<int>(i) % perRow;
        const int y0 = cell + cap + r * (2 * cell + 2 * cap);
        toTile(filters[i], cell).copyTo(sheet(cv::Rect(c * cell, y0, cell, cell)));
        cv::putText(sheet, "filter " + std::to_string(i + 1), cv::Point(c * cell + 4, y0 + cell + 13), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        toTile(responses[i], cell).copyTo(sheet(cv::Rect(c * cell, y0 + cell + cap, cell, cell)));
        cv::putText(sheet, "response " + std::to_string(i + 1), cv::Point(c * cell + 4, y0 + 2 * cell + cap + 13), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    return sheet;
}

}  // namespace p5
