#include "colorquant.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "kmeans.h"

std::vector<cv::Vec3b> sampleColors(const cv::Mat& bgr, int B, std::mt19937& rng) {
    CV_Assert(bgr.type() == CV_8UC3 && B >= 1);
    std::vector<cv::Vec3b> data;
    std::uniform_int_distribution<int> jitter(0, B - 1);
    for (int i = 0; i + B <= bgr.rows; i += B) {
        for (int j = 0; j + B <= bgr.cols; j += B) {
            data.push_back(bgr.at<cv::Vec3b>(i + jitter(rng), j + jitter(rng)));
        }
    }
    return data;
}

cv::Vec3b closestColor(const cv::Vec3b& pix, const std::vector<cv::Vec3b>& colors) {
    if (colors.empty()) throw std::invalid_argument("closestColor: empty palette");
    size_t best = 0;
    int bestD = ssd3(pix, colors[0]);
    for (size_t i = 1; i < colors.size(); ++i) {
        const int d = ssd3(pix, colors[i]);
        if (d < bestD) {
            bestD = d;
            best = i;
        }
    }
    return colors[best];
}

cv::Mat quantize(const cv::Mat& bgr, const std::vector<cv::Vec3b>& palette) {
    CV_Assert(bgr.type() == CV_8UC3);
    cv::Mat dst(bgr.size(), bgr.type());
    cv::parallel_for_(cv::Range(0, bgr.rows), [&](const cv::Range& r) {
        for (int i = r.start; i < r.end; ++i) {
            for (int j = 0; j < bgr.cols; ++j) {
                dst.at<cv::Vec3b>(i, j) = closestColor(bgr.at<cv::Vec3b>(i, j), palette);
            }
        }
    });
    return dst;
}

std::vector<double> paletteShares(const cv::Mat& bgr, const std::vector<cv::Vec3b>& palette) {
    CV_Assert(bgr.type() == CV_8UC3 && !palette.empty());
    std::vector<double> counts(palette.size(), 0.0);
    for (int i = 0; i < bgr.rows; ++i) {
        for (int j = 0; j < bgr.cols; ++j) {
            const cv::Vec3b& p = bgr.at<cv::Vec3b>(i, j);
            size_t best = 0;
            int bestD = ssd3(p, palette[0]);
            for (size_t k = 1; k < palette.size(); ++k) {
                const int d = ssd3(p, palette[k]);
                if (d < bestD) {
                    bestD = d;
                    best = k;
                }
            }
            counts[best] += 1.0;
        }
    }
    const double total = static_cast<double>(bgr.total());
    for (double& c : counts) c /= total;
    return counts;
}

cv::Mat paletteStrip(const std::vector<cv::Vec3b>& palette, const std::vector<double>& shares,
                     int width, int height) {
    CV_Assert(!palette.empty() && width >= static_cast<int>(palette.size()));
    cv::Mat strip(height, width, CV_8UC3, cv::Scalar::all(0));
    const bool weighted = shares.size() == palette.size();
    double total = 0;
    if (weighted) {
        for (double s : shares) total += s;
    }
    int x = 0;
    double acc = 0;
    for (size_t k = 0; k < palette.size(); ++k) {
        int next;
        if (k + 1 == palette.size()) {
            next = width;
        } else if (weighted && total > 0) {
            acc += shares[k];
            next = static_cast<int>(width * acc / total + 0.5);
        } else {
            next = static_cast<int>(width * (k + 1) / palette.size());
        }
        next = std::max(next, x + 1);
        next = std::min(next, width - static_cast<int>(palette.size() - k - 1));
        strip(cv::Rect(x, 0, next - x, height)).setTo(cv::Scalar(palette[k][0], palette[k][1], palette[k][2]));
        x = next;
    }
    return strip;
}

std::string toHex(const cv::Vec3b& bgr) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "#%02X%02X%02X", bgr[2], bgr[1], bgr[0]);
    return buf;
}

double meanSquaredError(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.size() == b.size() && a.type() == b.type());
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    const cv::Scalar s = cv::sum(diff);
    return (s[0] + s[1] + s[2] + s[3]) / (static_cast<double>(a.total()) * a.channels());
}
