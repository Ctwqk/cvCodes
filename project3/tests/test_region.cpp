#include <map>

#include <opencv2/imgproc.hpp>

#include "doctest.h"
#include "region.h"

using cv::Mat;
using cv::Vec3b;

namespace {
Mat randomBinary(int rows, int cols, unsigned seed, double density) {
    Mat m = Mat::zeros(rows, cols, CV_8UC1);
    cv::RNG rng(seed);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (rng.uniform(0.0, 1.0) < density) m.at<uchar>(i, j) = 255;
    return m;
}
Mat blobImage(const std::vector<cv::Rect>& blobs, cv::Size size = cv::Size(200, 160)) {
    Mat m = Mat::zeros(size, CV_8UC1);
    for (const cv::Rect& r : blobs) m(r).setTo(255);
    return m;
}
}  // namespace

TEST_CASE("labelRegions is the same partition as cv::connectedComponents (4-connectivity)") {
    for (unsigned seed : {1u, 2u, 3u}) {
        for (double density : {0.3, 0.55}) {
            const Mat img = randomBinary(37, 53, seed, density);
            Mat labels, ref;
            std::vector<int> areas;
            const int n = labelRegions(img, labels, areas);
            const int nRef = cv::connectedComponents(img, ref, 4, CV_32S) - 1;
            REQUIRE(n == nRef);
            REQUIRE(areas.size() == static_cast<size_t>(n));
            // bijection between our labels and OpenCV's
            std::map<int, int> mapping;
            for (int i = 0; i < img.rows; i++)
                for (int j = 0; j < img.cols; j++) {
                    const int a = labels.at<int>(i, j), b = ref.at<int>(i, j);
                    REQUIRE((a == 0) == (b == 0));
                    if (!a) continue;
                    auto [it, inserted] = mapping.emplace(a, b);
                    REQUIRE(it->second == b);
                }
            std::map<int, int> inverse;
            for (auto& [a, b] : mapping) CHECK(inverse.emplace(b, a).second);
            int sum = 0;
            for (int a : areas) sum += a;
            CHECK(sum == cv::countNonZero(img));
        }
    }
}

TEST_CASE("labelRegions: diagonal pixels are NOT connected (4-connectivity)") {
    Mat img = Mat::zeros(4, 4, CV_8UC1);
    img.at<uchar>(1, 1) = 255;
    img.at<uchar>(2, 2) = 255;
    Mat labels;
    std::vector<int> areas;
    CHECK(labelRegions(img, labels, areas) == 2);
    CHECK(labelRegions(Mat(), labels, areas) == 0);
}

TEST_CASE("regionize returns the largest interior regions first, as separate masks") {
    const Mat img = blobImage({cv::Rect(20, 20, 40, 30), cv::Rect(100, 40, 50, 40), cv::Rect(60, 110, 40, 30)});
    Mat dst;
    const auto masks = regionize(img, dst);
    REQUIRE(masks.size() == 3);
    CHECK(cv::countNonZero(masks[0]) == 50 * 40);
    CHECK(cv::countNonZero(masks[1]) == 40 * 30);  // equal areas: the blob that starts first comes first
    CHECK(cv::countNonZero(masks[2]) == 40 * 30);
    CHECK(masks[0].at<uchar>(50, 120) == 255);
    CHECK(masks[0].at<uchar>(30, 30) == 0);  // masks are distinct, not aliases of one buffer
    CHECK(masks[1].at<uchar>(30, 30) == 255);
    CHECK(masks[2].at<uchar>(125, 80) == 255);
    CHECK(dst.type() == CV_8UC3);
    CHECK(dst.at<Vec3b>(50, 120) == Vec3b(255, 0, 0));  // first region blue
    CHECK(dst.at<Vec3b>(30, 30) == Vec3b(0, 255, 0));   // second green
    CHECK(dst.at<Vec3b>(125, 80) == Vec3b(0, 0, 255));  // third red
    CHECK(dst.at<Vec3b>(5, 5) == Vec3b(0, 0, 0));
}

TEST_CASE("regionize does not modify its input and the colour order continues cyan, yellow, magenta") {
    const Mat img = blobImage({cv::Rect(10, 10, 30, 40), cv::Rect(60, 10, 29, 40), cv::Rect(110, 10, 28, 40),
                               cv::Rect(10, 80, 27, 40), cv::Rect(60, 80, 26, 40), cv::Rect(110, 80, 25, 40)});
    const Mat copy = img.clone();
    Mat dst;
    const auto masks = regionize(img, dst);
    CHECK(cv::norm(img, copy, cv::NORM_INF) == 0);  // the original divided src by 255 in place
    REQUIRE(masks.size() == 6);
    CHECK(dst.at<Vec3b>(25, 25) == Vec3b(255, 0, 0));
    CHECK(dst.at<Vec3b>(95, 20) == Vec3b(255, 255, 0));    // 4th: cyan (B + G)
    CHECK(dst.at<Vec3b>(95, 70) == Vec3b(0, 255, 255));    // 5th: yellow (G + R)
    CHECK(dst.at<Vec3b>(95, 120) == Vec3b(255, 0, 255));   // 6th: magenta (R + B)
}

TEST_CASE("regionize drops regions touching the border band and regions that are too small") {
    Mat img = blobImage({cv::Rect(0, 50, 40, 40),      // touches the left edge
                         cv::Rect(80, 3, 30, 30),      // inside the 6 px band at the top
                         cv::Rect(140, 60, 20, 20),    // 400 px: too small
                         cv::Rect(60, 90, 40, 40)});   // fine
    Mat dst;
    auto masks = regionize(img, dst);
    REQUIRE(masks.size() == 1);
    CHECK(cv::countNonZero(masks[0]) == 1600);
    masks = regionize(img, dst, 6, 300);  // lower the area limit: the 20x20 blob now qualifies
    CHECK(masks.size() == 2);
    // a region whose pixels all start exactly kEdgeBand pixels in is kept
    Mat inner = blobImage({cv::Rect(kEdgeBand, kEdgeBand, 40, 40)});
    CHECK(regionize(inner, dst).size() == 1);
    Mat touching = blobImage({cv::Rect(kEdgeBand - 1, kEdgeBand, 40, 40)});
    CHECK(regionize(touching, dst).empty());
}

TEST_CASE("regionize limits: num is clamped to 6, 0 gives nothing, wrong input gives nothing") {
    std::vector<cv::Rect> blobs;
    for (int i = 0; i < 8; i++) blobs.push_back(cv::Rect(10 + 22 * i, 20, 20, 60));  // 8 blobs of 1200 px
    Mat img = blobImage(blobs, cv::Size(200, 100));
    Mat dst;
    CHECK(regionize(img, dst, 10).size() == 6);  // the original wrote past its 6 planes here
    CHECK(regionize(img, dst, 2).size() == 2);
    CHECK(regionize(img, dst, 0).empty());
    CHECK(regionize(Mat(), dst).empty());
    CHECK(regionize(Mat(10, 10, CV_8UC3), dst).empty());
}

TEST_CASE("regionize breaks ties by position and may alias src and dst") {
    Mat img = blobImage({cv::Rect(100, 20, 40, 30), cv::Rect(20, 20, 40, 30)});  // equal areas
    Mat dst;
    const auto masks = regionize(img, dst);
    REQUIRE(masks.size() == 2);
    CHECK(masks[0].at<uchar>(30, 30) == 255);  // left blob has the earlier first pixel
    Mat alias = img.clone();
    const auto again = regionize(alias, alias);
    CHECK(again.size() == 2);
    CHECK(alias.type() == CV_8UC3);
}
