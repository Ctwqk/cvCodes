#include "mnist.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace p5 {

cv::Mat makeBlob(const cv::Mat& gray28) {
    // (x - 127.5) / 127.5 == (x / 255 - 0.5) / 0.5
    return cv::dnn::blobFromImage(gray28, 1.0 / 127.5, cv::Size(28, 28), cv::Scalar(127.5), false, false, CV_32F);
}

Prediction predictDigit(cv::dnn::Net& net, const cv::Mat& gray28) {
    Prediction p;
    if (gray28.empty() || gray28.type() != CV_8UC1 || gray28.rows != 28 || gray28.cols != 28) return p;
    net.setInput(makeBlob(gray28));
    cv::Mat out = net.forward();  // 1x10 log-probabilities
    const cv::Mat row = out.reshape(1, 1);
    if (row.type() != CV_32F || row.cols != 10 || !cv::checkRange(row))
        throw std::runtime_error("MNIST model must output 10 finite float scores");
    double maxLog = -1e30;
    for (int i = 0; i < row.cols; i++) maxLog = std::max(maxLog, static_cast<double>(row.at<float>(0, i)));
    double sum = 0;
    p.probabilities.resize(static_cast<size_t>(row.cols));
    for (int i = 0; i < row.cols; i++) {
        p.probabilities[static_cast<size_t>(i)] = static_cast<float>(std::exp(row.at<float>(0, i) - maxLog));  // stable even if the net outputs raw logits
        sum += p.probabilities[static_cast<size_t>(i)];
    }
    int best = 0;
    for (size_t i = 0; i < p.probabilities.size(); i++) {
        p.probabilities[i] = static_cast<float>(p.probabilities[i] / sum);
        if (p.probabilities[i] > p.probabilities[static_cast<size_t>(best)]) best = static_cast<int>(i);
    }
    p.digit = best;
    p.probability = p.probabilities[static_cast<size_t>(best)];
    return p;
}

EvalResult evaluate(cv::dnn::Net& net, const std::vector<cv::Mat>& images, const std::vector<uint8_t>& labels) {
    EvalResult r;
    const size_t n = std::min(images.size(), labels.size());
    for (size_t i = 0; i < n; i++) {
        const Prediction p = predictDigit(net, images[i]);
        if (p.digit < 0 || labels[i] > 9) continue;
        r.total++;
        r.confusion[labels[i]][p.digit]++;
        if (p.digit == labels[i]) r.correct++;
        else r.wrong.push_back(i);
    }
    return r;
}

cv::Mat preprocessDigit(const cv::Mat& input, cv::Rect* digitBox) {
    if (input.empty()) return cv::Mat();
    cv::Mat gray;
    if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else if (input.channels() == 4) cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    else gray = input;
    if (gray.depth() != CV_8U) return cv::Mat();
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // ink = the minority class: dark strokes on light paper (the usual case) become white
    cv::Mat ink;
    cv::threshold(gray, ink, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    if (cv::countNonZero(ink) > static_cast<int>(ink.total() / 2)) cv::bitwise_not(ink, ink);

    // largest blob = the digit
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(ink, labels, stats, centroids, 8, CV_32S);
    int best = -1, bestArea = 0;
    for (int i = 1; i < n; i++) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) {
            bestArea = area;
            best = i;
        }
    }
    if (best < 0 || bestArea < 10) return cv::Mat();
    const cv::Rect box(stats.at<int>(best, cv::CC_STAT_LEFT), stats.at<int>(best, cv::CC_STAT_TOP),
                       stats.at<int>(best, cv::CC_STAT_WIDTH), stats.at<int>(best, cv::CC_STAT_HEIGHT));
    if (digitBox) *digitBox = box;
    cv::Mat mask = (labels(box) == best);

    // fit into 20x20 keeping the aspect ratio ...
    const double scale = 20.0 / std::max(box.width, box.height);
    cv::Mat small;
    cv::resize(mask, small, cv::Size(std::max(1, cvRound(box.width * scale)), std::max(1, cvRound(box.height * scale))), 0, 0,
               cv::INTER_AREA);
    // ... and place it so that its centre of mass is the centre of the 28x28 image (as MNIST was prepared)
    const cv::Moments m = cv::moments(small, true);
    const double cx = m.m00 > 0 ? m.m10 / m.m00 : small.cols / 2.0, cy = m.m00 > 0 ? m.m01 / m.m00 : small.rows / 2.0;
    const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 14.0 - cx, 0, 1, 14.0 - cy);
    cv::Mat canvas = cv::Mat::zeros(28, 28, CV_8UC1);
    cv::warpAffine(small, canvas, shift, cv::Size(28, 28), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return canvas;
}

cv::Mat predictionSheet(cv::dnn::Net& net, const std::vector<cv::Mat>& images, const std::vector<uint8_t>& labels,
                        const std::vector<size_t>& indices, int columns, int scale) {
    const int cell = 28 * scale, cap = 16;
    const int count = static_cast<int>(indices.size());
    const int rows = std::max(1, (count + columns - 1) / columns);
    cv::Mat sheet(rows * (cell + cap), columns * cell, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int k = 0; k < count; k++) {
        const size_t i = indices[static_cast<size_t>(k)];
        if (i >= images.size() || i >= labels.size()) continue;
        const Prediction p = predictDigit(net, images[i]);
        cv::Mat big, bgr;
        cv::resize(images[i], big, cv::Size(cell, cell), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(big, bgr, cv::COLOR_GRAY2BGR);
        const int r = k / columns, c = k % columns;
        bgr.copyTo(sheet(cv::Rect(c * cell, r * (cell + cap), cell, cell)));
        char text[32];
        std::snprintf(text, sizeof text, "%d > %d", labels[i], p.digit);
        cv::putText(sheet, text, cv::Point(c * cell + 4, r * (cell + cap) + cell + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    p.digit == labels[i] ? cv::Scalar(0, 120, 0) : cv::Scalar(0, 0, 220), 1, cv::LINE_AA);
    }
    return sheet;
}

}  // namespace p5
