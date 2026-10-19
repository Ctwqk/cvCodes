#include "imgfeatures.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {
bool isBGR8(const cv::Mat& m) { return !m.empty() && m.type() == CV_8UC3; }

// chromaticity bin of one pixel: x from green, y from red (BGR order)
inline int chromaBin(uchar b, uchar g, uchar r) {
    const float sum = static_cast<float>(b) + g + r;
    int x = 0, y = 0;
    if (sum > 0) {
        x = static_cast<int>(g / sum * (kHistBins - 1) + 0.5f);
        y = static_cast<int>(r / sum * (kHistBins - 1) + 0.5f);
    }
    return x * kHistBins + y;
}

void accumulateHistogram(const cv::Mat& bgr, float* bins) {
    for (int i = 0; i < bgr.rows; i++) {
        const uchar* p = bgr.ptr<uchar>(i);
        for (int j = 0; j < bgr.cols; j++) bins[chromaBin(p[3 * j], p[3 * j + 1], p[3 * j + 2])]++;
    }
}
}  // namespace

std::vector<float> mat2vec(const cv::Mat& m) {
    cv::Mat f;
    m.convertTo(f, CV_32F);
    f = f.reshape(1, 1).clone();  // 1 x (rows*cols*channels), continuous
    return std::vector<float>(f.ptr<float>(0), f.ptr<float>(0) + f.cols);
}

int central7x7(const cv::Mat& bgr, cv::Mat& patch) {
    if (!isBGR8(bgr) || bgr.rows < 7 || bgr.cols < 7) return -1;
    patch = bgr(cv::Rect(bgr.cols / 2 - 3, bgr.rows / 2 - 3, 7, 7)).clone();
    return 0;
}

int histogram(const cv::Mat& bgr, std::vector<float>& feature) {
    if (!isBGR8(bgr)) return -1;
    feature.assign(kHistSize, 0.f);
    accumulateHistogram(bgr, feature.data());
    return 0;
}

int multiHist(const cv::Mat& bgr, std::vector<float>& feature) {
    const int s = kMultiHistRoiSize;
    if (!isBGR8(bgr) || bgr.rows < s || bgr.cols < s) return -1;
    const int row = bgr.rows, col = bgr.cols;
    const cv::Rect rois[kMultiHistRois] = {
        cv::Rect(col / 2 - s / 2, row / 2 - s / 2, s, s),  // centre
        cv::Rect(0, row - s, s, s),                        // bottom left
        cv::Rect(col - s, 0, s, s),                        // top right
        cv::Rect(col - s, row - s, s, s),                  // bottom right
        cv::Rect(0, 0, s, s),                              // top left
    };
    feature.assign(static_cast<size_t>(kMultiHistRois) * kHistSize, 0.f);
    for (int r = 0; r < kMultiHistRois; r++) {
        accumulateHistogram(bgr(rois[r]), feature.data() + static_cast<size_t>(r) * kHistSize);
    }
    return 0;
}

int textureBinIndex(float angleDeg, float mag, float lo, float hi) {
    const int a = std::max(0, std::min(static_cast<int>(angleDeg / 15.f), kTextureAngleBins - 1));
    // The original divided by (max - min) without subtracting min, which can push the
    // magnitude bin past 31 and, in the last angle row, out of the feature vector.
    const double m = hi > lo ? (static_cast<double>(mag) - lo) / (static_cast<double>(hi) - lo) : 0.0;
    const int b = std::max(0, std::min(static_cast<int>(m * (kHistBins - 1) + 0.5), kHistBins - 1));
    return a * kHistBins + b;
}

int texture(const cv::Mat& bgr, std::vector<float>& feature) {
    if (!isBGR8(bgr)) return -1;
    cv::Mat f32, sx, sy;
    bgr.convertTo(f32, CV_32FC3);
    cv::Sobel(f32, sx, CV_32F, 1, 0, 3);
    cv::Sobel(f32, sy, CV_32F, 0, 1, 3);
    cv::Mat mag, ang;
    cv::magnitude(sx.reshape(1), sy.reshape(1), mag);  // per channel value
    ang.create(mag.size(), CV_32F);
    {
        const cv::Mat x1 = sx.reshape(1), y1 = sy.reshape(1);
        for (int i = 0; i < ang.rows; i++) {
            const float* xp = x1.ptr<float>(i);
            const float* yp = y1.ptr<float>(i);
            float* ap = ang.ptr<float>(i);
            // orientation in degrees, moved from [-180, 180] to [0, 360]
            for (int j = 0; j < ang.cols; j++) ap[j] = 180.f + std::atan2(yp[j], xp[j]) * 180.f / static_cast<float>(CV_PI);
        }
    }
    double lo = 0, hi = 0;
    cv::minMaxLoc(mag, &lo, &hi);
    feature.assign(kTextureSize, 0.f);
    for (int i = 0; i < mag.rows; i++) {
        const float* mp = mag.ptr<float>(i);
        const float* ap = ang.ptr<float>(i);
        for (int j = 0; j < mag.cols; j++) {
            feature[static_cast<size_t>(textureBinIndex(ap[j], mp[j], static_cast<float>(lo), static_cast<float>(hi)))]++;
        }
    }
    return 0;
}

int gaborHist(const cv::Mat& bgr, std::vector<float>& feature) {
    if (!isBGR8(bgr)) return -1;
    const cv::Mat kernel = cv::getGaborKernel(cv::Size(30, 30), 1, 0, 1, 0.02);
    cv::Mat gray, filtered, u8;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);
    cv::filter2D(gray, filtered, CV_32F, kernel);
    double lo = 0, hi = 0;
    cv::minMaxLoc(filtered, &lo, &hi);
    if (hi > lo) {
        filtered.convertTo(u8, CV_8U, 255.0 / (hi - lo), -255.0 * lo / (hi - lo));
    } else {
        u8 = cv::Mat::zeros(filtered.size(), CV_8U);  // flat response: everything in bin 0
    }
    feature.assign(kGaborSize, 0.f);
    for (int i = 0; i < u8.rows; i++) {
        const uchar* p = u8.ptr<uchar>(i);
        for (int j = 0; j < u8.cols; j++) feature[p[j] / 8]++;
    }
    return 0;
}

int fourierHist(const cv::Mat& bgr, std::vector<float>& feature) {
    const int B = kFourierBlock;
    if (!isBGR8(bgr) || bgr.rows < 2 * B || bgr.cols < 2 * B) return -1;
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    const int row = cv::getOptimalDFTSize(gray.rows), col = cv::getOptimalDFTSize(gray.cols);
    cv::copyMakeBorder(gray, gray, 0, row - gray.rows, 0, col - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(gray), cv::Mat::zeros(gray.size(), CV_32F)};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg);
    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::log(planes[0] + 1, planes[0]);
    cv::normalize(planes[0], planes[0], 0, 1, cv::NORM_MINMAX);

    // The spectrum is not shifted, so the DC component sits in the top-left corner and the
    // lowest frequencies in the four corners of the matrix: take an 8x8 block from each.
    // (The original had `float TIME = 1/4`, i.e. 0 by integer division, which selected exactly
    // these corner blocks; this makes that explicit.)
    const cv::Mat& s = planes[0];
    const cv::Rect blocks[4] = {cv::Rect(0, 0, B, B), cv::Rect(0, row - B, B, B),
                                cv::Rect(col - B, 0, B, B), cv::Rect(col - B, row - B, B, B)};
    cv::Mat joined;
    cv::hconcat(std::vector<cv::Mat>{s(blocks[0]), s(blocks[1]), s(blocks[2]), s(blocks[3])}, joined);
    feature = mat2vec(joined);
    for (float& v : feature) v *= kDnnMaxValue;
    return 0;
}
