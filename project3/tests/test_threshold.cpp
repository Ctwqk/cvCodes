#include <cmath>

#include <opencv2/imgproc.hpp>

#include "doctest.h"
#include "threshold.h"

using cv::Mat;
using cv::Vec3b;

namespace {
// two grey populations (80 and 200 with noise)
Mat bimodal(int rows, int cols, unsigned seed) {
    Mat m(rows, cols, CV_8UC1);
    cv::RNG rng(seed);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            const double base = (j < cols / 3) ? 200.0 : 80.0;
            m.at<uchar>(i, j) = cv::saturate_cast<uchar>(base + rng.gaussian(8));
        }
    return m;
}
}  // namespace

TEST_CASE("otsuThreshold agrees with cv::threshold(THRESH_OTSU)") {
    for (unsigned seed : {1u, 2u, 3u}) {
        const Mat g = bimodal(60, 90, seed);
        int pos = -1;
        const float variance = otsuThreshold(g, pos);
        Mat ref;
        const double t = cv::threshold(g, ref, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        CHECK(pos == static_cast<int>(t));
        CHECK(variance > 0.f);
        Mat mine;
        REQUIRE(myThreshold(g, mine, pos, 255) == 0);
        CHECK(cv::norm(mine, ref, cv::NORM_INF) == 0);  // pixels > t are foreground, like cv::threshold
        CHECK(pos > 90);
        CHECK(pos < 190);
    }
}

TEST_CASE("otsuThreshold on flat or degenerate images: no NaN, nothing ends up above the threshold") {
    Mat flat(10, 10, CV_8UC1, cv::Scalar(77));
    int pos = 5;
    const float v = otsuThreshold(flat, pos);
    CHECK(v == 0.f);
    CHECK(pos == 77);
    CHECK(std::isfinite(v));
    Mat bin;
    REQUIRE(myThreshold(flat, bin, pos, 255) == 0);
    CHECK(cv::countNonZero(bin) == 0);
    CHECK(otsuThreshold(Mat(), pos) == 0.f);
    Mat colour(4, 4, CV_8UC3);
    CHECK(otsuThreshold(colour, pos) == 0.f);
}

TEST_CASE("kmeanThreshold lands between the two modes and handles flat images") {
    const Mat g = bimodal(60, 90, 4);
    int pos = 0;
    const float t = kmeanThreshold(g, pos);
    CHECK(t > 100.f);
    CHECK(t < 180.f);
    CHECK(pos == static_cast<int>(t));
    Mat flat(8, 8, CV_8UC1, cv::Scalar(100));
    CHECK(kmeanThreshold(flat, pos) == doctest::Approx(100.f));
    CHECK(pos == 100);
}

TEST_CASE("averageThreshold is the mean (the original summed into an uninitialised float)") {
    Mat g(2, 2, CV_8UC1);
    g.at<uchar>(0, 0) = 10;
    g.at<uchar>(0, 1) = 20;
    g.at<uchar>(1, 0) = 30;
    g.at<uchar>(1, 1) = 41;
    int pos = 0;
    for (int i = 0; i < 3; i++) {  // repeatable
        CHECK(averageThreshold(g, pos) == doctest::Approx(25.25f));
        CHECK(pos == 25);
    }
}

TEST_CASE("myThreshold is strict and validates its input") {
    Mat g(1, 3, CV_8UC1);
    g.at<uchar>(0, 0) = 99;
    g.at<uchar>(0, 1) = 100;
    g.at<uchar>(0, 2) = 101;
    Mat out;
    REQUIRE(myThreshold(g, out, 100, 255) == 0);
    CHECK(out.at<uchar>(0, 0) == 0);
    CHECK(out.at<uchar>(0, 1) == 0);
    CHECK(out.at<uchar>(0, 2) == 255);
    REQUIRE(myThreshold(g, out, 0, 7) == 0);  // threshold 0 is allowed, upper value is used
    CHECK(out.at<uchar>(0, 0) == 7);
    CHECK(myThreshold(Mat(), out, 1, 255) == -1);
}

TEST_CASE("toBinary: bright object on dark background becomes white") {
    Mat img(60, 80, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(img, cv::Rect(20, 15, 30, 25), cv::Scalar(220, 220, 220), cv::FILLED);
    Mat bin;
    REQUIRE(toBinary(img, bin) == 0);
    REQUIRE(bin.type() == CV_8UC1);
    CHECK(bin.at<uchar>(25, 30) == 255);
    CHECK(bin.at<uchar>(2, 2) == 0);
    CHECK(cv::countNonZero(bin) == 30 * 25);
}

TEST_CASE("toBinary: explicit plane selection and bad input") {
    Mat img(40, 40, CV_8UC3, cv::Scalar(10, 10, 10));
    img(cv::Rect(10, 10, 10, 10)).setTo(cv::Scalar(250, 10, 10));  // only the blue plane sees the square
    Mat blue, red;
    REQUIRE(toBinary(img, blue, 0) == 0);
    CHECK(cv::countNonZero(blue) == 100);
    REQUIRE(toBinary(img, red, 2) == 0);  // red plane is flat: nothing above the threshold
    CHECK(cv::countNonZero(red) == 0);
    Mat automatic;
    REQUIRE(toBinary(img, automatic) == 0);
    CHECK(cv::countNonZero(automatic) == 100);
    Mat out;
    CHECK(toBinary(Mat(), out) == -1);
    CHECK(toBinary(Mat(5, 5, CV_8UC1), out) == -1);
    CHECK(toBinary(img, out, 7) == -1);
}
