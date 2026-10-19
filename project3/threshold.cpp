#include "threshold.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {
void histogram256(const cv::Mat& gray, double hist[256]) {
    std::fill(hist, hist + 256, 0.0);
    for (int i = 0; i < gray.rows; i++) {
        const uchar* p = gray.ptr<uchar>(i);
        for (int j = 0; j < gray.cols; j++) hist[p[j]] += 1.0;
    }
}
}  // namespace

// Otsu's method: find the threshold that maximises the inter-class variance.
float otsuThreshold(const cv::Mat& gray, int& pos) {
    pos = 0;
    if (gray.empty() || gray.type() != CV_8UC1) return 0.f;
    double hist[256];
    histogram256(gray, hist);
    const double total = static_cast<double>(gray.total());
    double sumAll = 0;
    for (int i = 0; i < 256; i++) sumAll += i * hist[i];

    // running sums make every candidate threshold O(1)
    double wl = 0, sl = 0, best = 0;
    for (int t = 0; t < 255; t++) {
        wl += hist[t];
        sl += t * hist[t];
        const double wr = total - wl;
        if (wl == 0 || wr == 0) continue;  // one class is empty: no valid split (used to give 0/0)
        const double mu = sl / wl - (sumAll - sl) / wr;
        const double var = mu * mu * (wl / total) * (wr / total);
        if (var > best) {
            best = var;
            pos = t;
        }
    }
    if (best == 0) {  // flat image: no split exists. Put the threshold at the top level so nothing is "upper".
        double lo = 0, hi = 0;
        cv::minMaxLoc(gray, &lo, &hi);
        pos = static_cast<int>(hi);
    }
    return static_cast<float>(best);
}

// Use two-means clustering on the grey levels; the threshold is the midpoint of the two centroids.
float kmeanThreshold(const cv::Mat& gray, int& pos) {
    pos = 0;
    if (gray.empty() || gray.type() != CV_8UC1) return 0.f;
    double hist[256];
    histogram256(gray, hist);
    double t = 0;
    for (int i = 0; i < 256; i++) t += i * hist[i];
    t /= static_cast<double>(gray.total());  // start at the mean
    for (int iter = 0; iter < 256; iter++) {
        double nl = 0, sl = 0, nu = 0, su = 0;
        for (int i = 0; i < 256; i++) {
            if (i > t) {
                nu += hist[i];
                su += i * hist[i];
            } else {
                nl += hist[i];
                sl += i * hist[i];
            }
        }
        if (nl == 0 || nu == 0) break;  // flat image: one cluster only
        const double next = (sl / nl + su / nu) / 2;
        const bool converged = std::fabs(next - t) < 0.5;
        t = next;
        if (converged) break;
    }
    pos = static_cast<int>(t);
    return static_cast<float>(t);
}

float averageThreshold(const cv::Mat& gray, int& pos) {
    pos = 0;
    if (gray.empty() || gray.type() != CV_8UC1) return 0.f;
    const double mean = cv::mean(gray)[0];  // the original summed into an uninitialised float
    pos = static_cast<int>(mean);
    return static_cast<float>(mean);
}

int myThreshold(const cv::Mat& src, cv::Mat& dst, int threshold, int upper) {
    if (src.empty() || src.type() != CV_8UC1) return -1;
    cv::Mat out(src.size(), CV_8UC1, cv::Scalar(0));
    const uchar up = cv::saturate_cast<uchar>(upper);
    for (int i = 0; i < src.rows; i++) {
        const uchar* s = src.ptr<uchar>(i);
        uchar* d = out.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            if (s[j] > threshold) d[j] = up;
        }
    }
    dst = out;
    return 0;
}

int toBinary(const cv::Mat& src, cv::Mat& dst, int idx) {
    if (src.empty() || src.type() != CV_8UC3 || idx > 3) return -1;
    cv::Mat planes[4];
    cv::split(src, planes);
    cv::cvtColor(src, planes[3], cv::COLOR_BGR2GRAY);
    float variance[4] = {};
    int pos[4] = {};
    // choose the plane that gives the largest inter-class variance
    for (int i = 0; i < 4; i++) variance[i] = otsuThreshold(planes[i], pos[i]);
    // in practice the inter-class variance of the greyscale plane should be doubled
    variance[3] *= 2;
    if (idx < 0) idx = static_cast<int>(std::max_element(variance, variance + 4) - variance);
    return myThreshold(planes[idx], dst, pos[idx], 255);
}
