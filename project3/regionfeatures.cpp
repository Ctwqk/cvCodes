/*
 * Author: Taiwei Cui
 * Time: Feb 27 2024
 *
 * Portfolio upgrade: pixel coordinates live in heap vectors instead of
 * `int x[row*col]` on the stack (a 2 MB-plus stack frame per call), centred
 * coordinates keep their fractional part, the box measures pixel extents
 * (max - min + 1) and an empty region no longer divides by zero.
 */
#include "regionfeatures.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "csv_util.h"

std::vector<float> featureBox(const cv::Mat& map, cv::RotatedRect& ans, const std::string& savePath,
                              const std::string& label) {
    ans = cv::RotatedRect();
    if (map.empty() || map.type() != CV_8UC1) return {};
    std::vector<float> xs, ys;
    double xSum = 0, ySum = 0;
    for (int i = 0; i < map.rows; i++) {
        const uchar* p = map.ptr<uchar>(i);
        for (int j = 0; j < map.cols; j++) {
            if (p[j]) {
                xs.push_back(static_cast<float>(j));
                ys.push_back(static_cast<float>(i));
                xSum += j;
                ySum += i;
            }
        }
    }
    const size_t N = xs.size();
    if (N == 0) return {};
    const double xMean = xSum / N, yMean = ySum / N;

    // second central moments
    double u11 = 0, u02 = 0, u20 = 0;
    for (size_t i = 0; i < N; i++) {
        const double dx = xs[i] - xMean, dy = ys[i] - yMean;
        u11 += dx * dy;
        u20 += dx * dx;
        u02 += dy * dy;
    }
    u11 /= N;
    u20 /= N;
    u02 /= N;

    // rotation that makes X the minor axis and Y the major axis (axis of least moment)
    const double theta = 0.5 * std::atan2(2 * u11, u02 - u20);
    const double vx = std::cos(theta), vy = std::sin(theta);

    // extents in the rotated basis: X runs across the region (minor axis), Y along it (major axis)
    double maxX = -1e30, maxY = -1e30, minX = 1e30, minY = 1e30;
    for (size_t i = 0; i < N; i++) {
        const double dx = xs[i] - xMean, dy = ys[i] - yMean;
        const double tx = vx * dx - vy * dy;
        const double ty = vy * dx + vx * dy;
        maxX = std::max(maxX, tx);
        minX = std::min(minX, tx);
        maxY = std::max(maxY, ty);
        minY = std::min(minY, ty);
    }
    const double cx = (maxX + minX) / 2, cy = (maxY + minY) / 2;
    // centre of the box in image coordinates
    const double centreX = xMean + vx * cx + vy * cy;
    const double centreY = yMean + vx * cy - vy * cx;
    const double w = maxX - minX + 1, h = maxY - minY + 1;  // +1: extents of pixel centres cover whole pixels
    ans = cv::RotatedRect(cv::Point2f(static_cast<float>(centreX), static_cast<float>(centreY)),
                          cv::Size2f(static_cast<float>(w), static_cast<float>(h)),
                          static_cast<float>(-theta * 180.0 / CV_PI));

    std::vector<float> feature = {static_cast<float>(N / (w * h)), static_cast<float>(w / h),
                                  static_cast<float>(std::sqrt(static_cast<double>(N)) / (w + h))};
    if (!savePath.empty() || !label.empty()) {
        std::string err;
        if (!cvc::appendFeatureRow(savePath, label, feature, false, &err)) {
            std::fprintf(stderr, "featureBox: %s\n", err.c_str());
        }
    }
    return feature;
}

int drawBox(const cv::RotatedRect& box, const std::vector<float>& features, cv::Mat& dst, const std::string& name,
            int thickness) {
    if (dst.empty()) return -1;
    cv::Point2f vertices[4];
    box.points(vertices);
    cv::Scalar s;
    if (dst.channels() == 1) s = cv::Scalar(255);
    else s = cv::Scalar(255, 0, 255);
    const double angle = box.angle / 180.0 * CV_PI;
    cv::line(dst, box.center,
             cv::Point2f(box.center.x + static_cast<float>(std::sin(angle)) * 30, box.center.y - static_cast<float>(std::cos(angle)) * 30),
             cv::Scalar(255, 255, 255), thickness);
    for (int i = 0; i < 4; i++) cv::line(dst, vertices[i], vertices[(i + 1) % 4], s, thickness);
    if (!features.empty()) {
        char text[32];
        std::snprintf(text, sizeof text, "%.3f", static_cast<double>(features.back()));
        cv::putText(dst, text, vertices[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, s, 2);
    }
    if (!name.empty()) cv::putText(dst, name, vertices[3], cv::FONT_HERSHEY_SIMPLEX, 0.5, s, 2);
    return 0;
}
