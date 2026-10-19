#include <opencv2/imgproc.hpp>

#include "denoise.h"
#include "doctest.h"

using cv::Mat;

namespace {
// The original implementation (cv::Mat operations per pixel), applied to a copy of the input.
Mat referenceCleanUp(const Mat& input, int kernelSize, int operationType, int kernelType) {
    Mat src = input.clone();
    Mat kernel = Mat::ones(cv::Size(kernelSize, kernelSize), CV_8U);
    Mat dst = src.clone();
    if (!kernelType) {
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                if ((i % 2) ^ (j % 2)) kernel.at<uchar>(i, j) = 0;
    }
    kernel.at<uchar>(kernelSize / 2, kernelSize / 2) = 0;
    src /= 255;
    const int row = src.rows, col = src.cols;
    const int threshold = static_cast<int>(static_cast<float>(kernelSize) * kernelSize / 4 / (1 + kernelType));
    Mat tmp = Mat::ones(kernel.size(), CV_8U);
    tmp *= operationType;
    Mat res;
    for (int i = 0; i <= row - kernelSize; i++) {
        uchar* ptr = src.ptr<uchar>(i + kernelSize / 2);
        uchar* ptr1 = dst.ptr<uchar>(i + kernelSize / 2);
        for (int j = 0; j <= col - kernelSize; j++) {
            if (ptr[j + kernelSize / 2] != operationType) continue;
            cv::bitwise_xor(tmp, src(cv::Rect(j, i, kernelSize, kernelSize)), res);
            cv::bitwise_and(res, kernel, res);
            if (cv::sum(res)[0] > threshold) ptr1[j + kernelSize / 2] = static_cast<uchar>(1 - operationType);
        }
    }
    dst *= 255;
    return dst;
}

Mat randomBinary(int rows, int cols, unsigned seed, double density) {
    Mat m = Mat::zeros(rows, cols, CV_8UC1);
    cv::RNG rng(seed);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (rng.uniform(0.0, 1.0) < density) m.at<uchar>(i, j) = 255;
    return m;
}
}  // namespace

TEST_CASE("cleanUp is identical to the original implementation for every kernel/operation") {
    for (unsigned seed : {1u, 2u}) {
        const Mat img = randomBinary(33, 41, seed, 0.45);
        for (int k : {3, 5})
            for (int op : {0, 1})
                for (int type : {0, 1}) {
                    Mat out;
                    REQUIRE(cleanUp(img, out, k, op, type) == 0);
                    INFO("k=" << k << " op=" << op << " type=" << type);
                    CHECK(cv::norm(out, referenceCleanUp(img, k, op, type), cv::NORM_INF) == 0);
                }
    }
}

TEST_CASE("cleanUp leaves its input untouched (the original divided it by 255 in place)") {
    const Mat img = randomBinary(20, 20, 3, 0.5);
    const Mat copy = img.clone();
    Mat out;
    REQUIRE(cleanUp(img, out, 3, 1, 0) == 0);
    CHECK(cv::norm(img, copy, cv::NORM_INF) == 0);
    REQUIRE(denoise(img, out) == 0);
    CHECK(cv::norm(img, copy, cv::NORM_INF) == 0);
}

TEST_CASE("cleanUp validates its arguments") {
    Mat img = randomBinary(10, 10, 1, 0.5), out;
    CHECK(cleanUp(img, out, 4, 1, 0) == -1);  // even kernel
    CHECK(cleanUp(img, out, 3, 2, 0) == -1);  // unknown operation
    CHECK(cleanUp(Mat(), out, 3, 1, 0) == -1);
    CHECK(cleanUp(Mat(5, 5, CV_8UC3), out, 3, 1, 0) == -1);
}

TEST_CASE("shrink removes isolated pixels, keeps solid blocks and the border") {
    Mat img = Mat::zeros(30, 30, CV_8UC1);
    img.at<uchar>(5, 5) = 255;                               // speckle
    img(cv::Rect(12, 12, 12, 12)).setTo(255);                // solid block
    img.at<uchar>(0, 0) = 255;                               // border pixel is never touched
    Mat out;
    REQUIRE(cleanUp(img, out, 3, 1, 0) == 0);
    CHECK(out.at<uchar>(5, 5) == 0);
    CHECK(out.at<uchar>(17, 17) == 255);
    CHECK(out.at<uchar>(12, 17) == 255);  // straight edge survives
    CHECK(out.at<uchar>(0, 0) == 255);
}

TEST_CASE("grow fills a pin-hole inside a region") {
    Mat img = Mat::zeros(30, 30, CV_8UC1);
    img(cv::Rect(8, 8, 14, 14)).setTo(255);
    img.at<uchar>(15, 15) = 0;
    Mat out;
    REQUIRE(cleanUp(img, out, 3, 0, 1) == 0);
    CHECK(out.at<uchar>(15, 15) == 255);
}

TEST_CASE("denoise removes salt noise, fills pepper holes, keeps the object, and may alias") {
    Mat img = Mat::zeros(120, 160, CV_8UC1);
    cv::circle(img, cv::Point(80, 60), 30, cv::Scalar(255), cv::FILLED);
    const int objectPixels = cv::countNonZero(img);
    cv::RNG rng(7);
    for (int n = 0; n < 150; n++) {  // salt outside, pepper inside
        const int x = rng.uniform(10, 150), y = rng.uniform(10, 110);
        const bool inside = (x - 80) * (x - 80) + (y - 60) * (y - 60) < 25 * 25;
        img.at<uchar>(y, x) = inside ? 0 : 255;
    }
    Mat out;
    REQUIRE(denoise(img, out) == 0);
    const Mat object = [&] { Mat m = Mat::zeros(img.size(), CV_8UC1); cv::circle(m, cv::Point(80, 60), 30, cv::Scalar(255), cv::FILLED); return m; }();
    // The grow passes of the original algorithm swell objects by ~2 px, so compare with the disc
    // dilated by 4 px: everything in `out` must lie inside it, and the disc must be fully covered.
    Mat wide, covered;
    cv::dilate(object, wide, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    Mat outside = out & ~wide;
    CHECK(cv::countNonZero(outside) == 0);                         // no salt survives
    cv::bitwise_and(out, object, covered);
    CHECK(cv::countNonZero(covered) == objectPixels);              // pepper holes are filled

    Mat inPlace = img.clone();
    REQUIRE(denoise(inPlace, inPlace) == 0);
    CHECK(cv::norm(inPlace, out, cv::NORM_INF) == 0);
}
