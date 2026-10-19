#include <opencv2/imgproc.hpp>

#include "doctest.h"
#include "pipeline.h"

using cv::Mat;

namespace {
// bright ellipse + rotated bright rectangle on a dark noisy background
Mat scene(bool darkObjects, unsigned seed = 3) {
    Mat img(480, 640, CV_8UC3, cv::Scalar(50, 55, 60));
    cv::ellipse(img, cv::Point(160, 240), cv::Size(70, 45), 0, 0, 360, cv::Scalar(215, 220, 225), cv::FILLED);
    cv::Point2f p[4];
    cv::RotatedRect(cv::Point2f(450, 250), cv::Size2f(170, 60), 35.f).points(p);
    std::vector<cv::Point> poly(p, p + 4);
    cv::fillConvexPoly(img, poly, cv::Scalar(205, 215, 220));
    Mat noise(img.size(), CV_8UC3);
    cv::RNG rng(seed);
    rng.fill(noise, cv::RNG::NORMAL, 0, 4);
    cv::add(img, noise, img);
    if (darkObjects) cv::bitwise_not(img, img);
    return img;
}
}  // namespace

TEST_CASE("segmentImage finds both objects and measures their shape") {
    SegmentResult r;
    REQUIRE(segmentImage(scene(false), SegmentOptions(), r) == 0);
    CHECK_FALSE(r.inverted);
    REQUIRE(r.regions.size() == 2);
    // largest first: ellipse area = pi*70*45 = 9896, rectangle 170*60 = 10200 -> rectangle first
    const RegionInfo& rect = r.regions[0];
    const RegionInfo& ell = r.regions[1];
    // denoise() dilates objects by about two pixels, hence the generous tolerance on the areas
    CHECK(rect.area == doctest::Approx(10200).epsilon(0.2));
    CHECK(ell.area == doctest::Approx(9896).epsilon(0.2));
    CHECK(rect.area > ell.area);
    CHECK(rect.features[1] == doctest::Approx(60.0 / 170.0).epsilon(0.06));
    CHECK(rect.features[0] > 0.9f);
    CHECK(ell.features[0] == doctest::Approx(CV_PI / 4).epsilon(0.05));
    CHECK(ell.features[1] == doctest::Approx(45.0 / 70.0).epsilon(0.06));
    CHECK(rect.box.center.x == doctest::Approx(450).epsilon(0.02));
    CHECK(ell.box.center.x == doctest::Approx(160).epsilon(0.02));
    CHECK(r.binary.size() == r.resized.size());
    CHECK(r.colored.type() == CV_8UC3);
}

TEST_CASE("segmentImage: automatic polarity gives the same objects for dark-on-bright images") {
    SegmentResult bright, dark, forced;
    REQUIRE(segmentImage(scene(false), SegmentOptions(), bright) == 0);
    REQUIRE(segmentImage(scene(true), SegmentOptions(), dark) == 0);
    CHECK(dark.inverted);
    REQUIRE(dark.regions.size() == bright.regions.size());
    for (size_t i = 0; i < bright.regions.size(); i++) {
        CHECK(dark.regions[i].area == doctest::Approx(bright.regions[i].area).epsilon(0.03));
        CHECK(dark.regions[i].features[1] == doctest::Approx(bright.regions[i].features[1]).epsilon(0.05));
    }
    // forcing the wrong polarity makes the background the object, which touches the border: nothing left
    SegmentOptions wrong;
    wrong.invert = 1;
    REQUIRE(segmentImage(scene(false), wrong, forced) == 0);
    CHECK(forced.regions.empty());
}

TEST_CASE("segmentImage: resizing keeps the aspect ratio, options are honoured, bad input rejected") {
    Mat big;
    cv::resize(scene(false), big, cv::Size(1600, 1200));
    SegmentResult r;
    REQUIRE(segmentImage(big, SegmentOptions(), r) == 0);
    CHECK(r.resized.size() == cv::Size(640, 480));
    CHECK(r.regions.size() == 2);

    SegmentOptions one;
    one.maxRegions = 1;
    REQUIRE(segmentImage(scene(false), one, r) == 0);
    CHECK(r.regions.size() == 1);

    SegmentOptions keep;
    keep.maxSide = 0;
    REQUIRE(segmentImage(big, keep, r) == 0);
    CHECK(r.resized.size() == big.size());

    SegmentOptions huge;
    huge.minArea = 100000;
    REQUIRE(segmentImage(scene(false), huge, r) == 0);
    CHECK(r.regions.empty());

    CHECK(segmentImage(Mat(), SegmentOptions(), r) == -1);
    CHECK(segmentImage(Mat(10, 10, CV_8UC1), SegmentOptions(), r) == -1);
}

TEST_CASE("annotate draws on a copy of the resized input") {
    SegmentResult r;
    REQUIRE(segmentImage(scene(false), SegmentOptions(), r) == 0);
    const Mat before = r.resized.clone();
    const Mat shown = annotate(r, {"rect", "ellipse"});
    CHECK(shown.size() == r.resized.size());
    CHECK(cv::norm(r.resized, before, cv::NORM_INF) == 0);
    CHECK(cv::norm(shown, before, cv::NORM_INF) > 0);
}
