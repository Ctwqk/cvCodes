#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "dnnEmbedding.h"
#include "doctest.h"

TEST_CASE("getEmbedding preserves aspect ratio for large and odd-sized object crops") {
    cv::dnn::Net net = cv::dnn::readNet(cvc::repoPath("project3/NetWorkThings/or2d-normmodel-007.onnx"));
    for (const cv::Size size : {cv::Size(200, 50), cv::Size(201, 50), cv::Size(50, 201), cv::Size(127, 53)}) {
        INFO("crop=" << size.width << "x" << size.height);
        cv::Mat object(size, CV_8UC1, cv::Scalar(255));
        cv::circle(object, cv::Point(size.width / 3, size.height / 2),
                   std::min(size.width, size.height) / 6, cv::Scalar(0), cv::FILLED);

        // Equivalent view of the same object already centered on a square black
        // canvas. getEmbedding adds its normal margin to this square reference.
        const int side = std::max(128, std::max(size.width, size.height));
        cv::Mat square = cv::Mat::zeros(side, side, CV_8UC1);
        object.copyTo(square(cv::Rect((side - size.width) / 2, (side - size.height) / 2,
                                      size.width, size.height)));
        cv::Mat actual, expected;
        REQUIRE(getEmbedding(object, actual, cv::Rect(0, 0, object.cols, object.rows), net) == 0);
        REQUIRE(getEmbedding(square, expected, cv::Rect(0, 0, side, side), net) == 0);
        REQUIRE(actual.total() == 50);
        REQUIRE(expected.size() == actual.size());
        CHECK(cv::norm(actual, expected, cv::NORM_INF) < 1e-5);
    }
}
