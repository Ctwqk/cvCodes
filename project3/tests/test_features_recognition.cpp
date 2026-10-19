#include <cmath>
#include <cstdio>
#include <filesystem>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "csv_util.h"
#include "dnnEmbedding.h"
#include "doctest.h"
#include "recognition.h"
#include "regionfeatures.h"

using cv::Mat;

namespace {
Mat rotatedRectMask(cv::Size size, cv::Point2f centre, cv::Size2f rect, float angle) {
    Mat m = Mat::zeros(size, CV_8UC1);
    cv::Point2f pts[4];
    cv::RotatedRect(centre, rect, angle).points(pts);
    std::vector<cv::Point> poly;
    for (const auto& p : pts) poly.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::fillConvexPoly(m, poly, cv::Scalar(255));
    return m;
}
}  // namespace

TEST_CASE("featureBox: axis aligned rectangle") {
    Mat m = Mat::zeros(100, 200, CV_8UC1);
    m(cv::Rect(30, 20, 100, 40)).setTo(255);
    cv::RotatedRect box;
    const auto f = featureBox(m, box);
    REQUIRE(f.size() == static_cast<size_t>(kShapeFeatureCount));
    CHECK(f[0] == doctest::Approx(1.0).epsilon(1e-3));                     // fills its box
    CHECK(f[1] == doctest::Approx(0.4).epsilon(1e-3));                     // short / long
    CHECK(f[2] == doctest::Approx(std::sqrt(4000.0) / 140.0).epsilon(1e-3));
    CHECK(box.center.x == doctest::Approx(79.5).epsilon(0.01));
    CHECK(box.center.y == doctest::Approx(39.5).epsilon(0.01));
    CHECK(std::max(box.size.width, box.size.height) == doctest::Approx(100.f).epsilon(0.01));
    CHECK(std::min(box.size.width, box.size.height) == doctest::Approx(40.f).epsilon(0.01));
}

TEST_CASE("featureBox features do not depend on the orientation (the box follows the principal axes)") {
    std::vector<std::vector<float>> all;
    for (float angle : {0.f, 20.f, 45.f, 90.f, 135.f, 170.f}) {
        const Mat m = rotatedRectMask(cv::Size(300, 300), cv::Point2f(150, 150), cv::Size2f(160, 50), angle);
        cv::RotatedRect box;
        const auto f = featureBox(m, box);
        REQUIRE(f.size() == 3);
        INFO("angle " << angle);
        CHECK(f[0] > 0.9f);  // pixel rounding only
        CHECK(f[0] <= 1.05f);
        CHECK(f[1] == doctest::Approx(50.0 / 160.0).epsilon(0.05));
        CHECK(box.center.x == doctest::Approx(150).epsilon(0.02));
        CHECK(box.center.y == doctest::Approx(150).epsilon(0.02));
        all.push_back(f);
    }
    for (const auto& f : all) {
        CHECK(f[0] == doctest::Approx(all[0][0]).epsilon(0.06));
        CHECK(f[1] == doctest::Approx(all[0][1]).epsilon(0.06));
        CHECK(f[2] == doctest::Approx(all[0][2]).epsilon(0.06));
    }
}

TEST_CASE("featureBox: discs fill pi/4 of their box and have aspect 1") {
    Mat m = Mat::zeros(200, 200, CV_8UC1);
    cv::circle(m, cv::Point(100, 100), 50, cv::Scalar(255), cv::FILLED);
    cv::RotatedRect box;
    const auto f = featureBox(m, box);
    REQUIRE(f.size() == 3);
    CHECK(f[0] == doctest::Approx(CV_PI / 4).epsilon(0.03));
    CHECK(f[1] == doctest::Approx(1.0).epsilon(0.03));
    CHECK(f[2] == doctest::Approx(0.4431).epsilon(0.03));  // sqrt(pi r^2) / 4r = sqrt(pi) / 4
}

TEST_CASE("featureBox: empty maps and single pixels do not divide by zero") {
    cv::RotatedRect box;
    CHECK(featureBox(Mat::zeros(10, 10, CV_8UC1), box).empty());
    CHECK(featureBox(Mat(), box).empty());
    CHECK(featureBox(Mat::ones(10, 10, CV_8UC3), box).empty());
    Mat one = Mat::zeros(10, 10, CV_8UC1);
    one.at<uchar>(4, 6) = 255;
    const auto f = featureBox(one, box);
    REQUIRE(f.size() == 3);
    for (float v : f) CHECK(std::isfinite(v));
    CHECK(f[0] == doctest::Approx(1.f));
}

TEST_CASE("featureBox handles regions larger than the old stack arrays could") {
    Mat big = Mat::zeros(2400, 3200, CV_8UC1);  // 7.7 M pixels: two int[row*col] stack arrays would be 61 MB
    big(cv::Rect(100, 100, 3000, 2200)).setTo(255);
    cv::RotatedRect box;
    const auto f = featureBox(big, box);
    REQUIRE(f.size() == 3);
    CHECK(f[0] == doctest::Approx(1.0).epsilon(1e-3));
}

TEST_CASE("featureBox appends to a CSV when a label is given") {
    const std::string path = (std::filesystem::temp_directory_path() / "cvc_test_p3_features.csv").string();
    std::remove(path.c_str());
    Mat m = Mat::zeros(50, 50, CV_8UC1);
    m(cv::Rect(10, 10, 20, 10)).setTo(255);
    cv::RotatedRect box;
    const auto f = featureBox(m, box, path, "rect");
    featureBox(m, box, path, "rect2");
    cvc::FeatureTable t;
    REQUIRE(cvc::readFeatureTable(path, t));
    REQUIRE(t.size() == 2);
    CHECK(t.names[0] == "rect");
    CHECK(t.data[0][1] == doctest::Approx(f[1]).epsilon(1e-5));
    std::remove(path.c_str());
}

TEST_CASE("drawBox draws something and rejects empty images") {
    Mat img = Mat::zeros(100, 100, CV_8UC3);
    cv::RotatedRect box(cv::Point2f(50, 50), cv::Size2f(40, 20), 30.f);
    CHECK(drawBox(box, {0.5f, 0.5f, 0.3f}, img, "x") == 0);
    CHECK(cv::countNonZero(img.reshape(1)) > 0);
    Mat empty;
    CHECK(drawBox(box, {}, empty) == -1);
}

// ---- recognition ----------------------------------------------------------

namespace {
cvc::FeatureTable toyDb() {
    cvc::FeatureTable db;
    // two features: first in [0, 1], second in the thousands (to need scaling)
    db.names = {"a:1", "a:2", "a:3", "b:1", "b:2", "b:3"};
    db.data = {{0.10f, 1000}, {0.12f, 1100}, {0.11f, 900}, {0.90f, 5000}, {0.88f, 5200}, {0.91f, 4900}};
    return db;
}
}  // namespace

TEST_CASE("labelOf / labelFromFilename") {
    CHECK(labelOf("ball") == "ball");
    CHECK(labelOf("ball:ball1.jpg") == "ball");
    CHECK(labelFromFilename("project3/trainset/ball12.jpg") == "ball");
    CHECK(labelFromFilename("spoon.jpg") == "spoon");
    CHECK(labelFromFilename("tri1.jpeg") == "tri");
    CHECK(labelFromFilename("a/b/balls1.jpg") == "balls");
    CHECK(labelFromFilename("noext7") == "noext");
}

TEST_CASE("columnStdDev with and without a skipped row") {
    cvc::FeatureTable t;
    t.names = {"x", "y", "z"};
    t.data = {{1, 10}, {3, 10}, {100, 10}};
    auto sd = columnStdDev(t);
    REQUIRE(sd.size() == 2);
    CHECK(sd[1] == 0.f);
    sd = columnStdDev(t, 2);
    CHECK(sd[0] == doctest::Approx(1.f));
    CHECK(columnStdDev(cvc::FeatureTable()).empty());
}

TEST_CASE("classify: nearest neighbour with every metric") {
    const auto db = toyDb();
    for (DistanceKind kind : {DistanceKind::Euclidean, DistanceKind::ScaledEuclidean}) {
        ClassifierConfig cfg;
        cfg.kind = kind;
        CHECK(classify({0.13f, 1050}, db, cfg).label == "a");
        CHECK(classify({0.89f, 5100}, db, cfg).label == "b");
    }
    ClassifierConfig cos;
    cos.kind = DistanceKind::Cosine;
    CHECK(classify({0.13f, 1050}, db, cos).label == "a");
}

TEST_CASE("classify: scaled Euclidean needs the scaling, plain Euclidean is dominated by the large feature") {
    const auto db = toyDb();
    // a point whose 2nd feature is closer to class b's range but whose 1st feature clearly says a
    const std::vector<float> q = {0.11f, 3200};
    ClassifierConfig plain, scaled;
    plain.kind = DistanceKind::Euclidean;
    scaled.kind = DistanceKind::ScaledEuclidean;
    CHECK(classify(q, db, plain).label == "b");   // 2nd feature dominates
    CHECK(classify(q, db, scaled).label == "a");  // both features weigh the same after scaling
}

TEST_CASE("classify: k-NN vote, tie goes to the closest neighbour, leave-one-out skip, empty/mismatched DB") {
    cvc::FeatureTable db;
    db.names = {"a:1", "b:1", "b:2", "a:2"};
    db.data = {{0.f}, {1.f}, {1.1f}, {5.f}};
    ClassifierConfig k1, k3, k4;
    k1.kind = k3.kind = k4.kind = DistanceKind::Euclidean;
    k3.k = 3;
    k4.k = 4;
    CHECK(classify({0.4f}, db, k1).label == "a");   // closest is a:1
    CHECK(classify({0.4f}, db, k3).label == "b");   // neighbours a:1, b:1, b:2 -> b wins 2:1
    CHECK(classify({0.4f}, db, k4).label == "a");   // 2:2, closest neighbour (a:1) decides
    const auto c = classify({0.f}, db, k1, {}, 0);  // row 0 skipped
    CHECK(c.label == "b");
    CHECK(c.neighbours == std::vector<size_t>{1});
    CHECK(classify({0.f}, cvc::FeatureTable(), k1).label.empty());
    CHECK(classify({0.f, 1.f}, db, k1).label.empty());  // no row has two values
}

TEST_CASE("leaveOneOut: separable clusters are perfect, overlapping ones are not, matrix sums up") {
    const auto db = toyDb();
    ClassifierConfig cfg;
    const ConfusionMatrix cm = leaveOneOut(db, cfg);
    REQUIRE(cm.labels == std::vector<std::string>{"a", "b"});
    CHECK(cm.total() == 6);
    CHECK(cm.correct() == 6);
    CHECK(cm.accuracy() == doctest::Approx(1.0));
    CHECK(cm.counts[0][0] == 3);

    cvc::FeatureTable mixed;
    mixed.names = {"a:1", "b:1", "a:2", "b:2"};
    mixed.data = {{0.f}, {0.1f}, {0.2f}, {0.3f}};  // alternating labels along a line
    cfg.kind = DistanceKind::Euclidean;
    const ConfusionMatrix bad = leaveOneOut(mixed, cfg);
    CHECK(bad.total() == 4);
    CHECK(bad.accuracy() < 1.0);
    CHECK(ConfusionMatrix().accuracy() == 0.0);
}

// ---- DNN embedding ---------------------------------------------------------

TEST_CASE("getEmbedding: 50 values, and it really uses the bounding box (the sample ignored it)") {
    const std::string model = cvc::repoPath("project3/NetWorkThings/or2d-normmodel-007.onnx");
    REQUIRE(std::filesystem::exists(model));
    cv::dnn::Net net = cv::dnn::readNet(model);

    Mat a = Mat::zeros(240, 320, CV_8UC1), b = Mat::zeros(240, 320, CV_8UC1);
    cv::ellipse(a, cv::Point(80, 90), cv::Size(40, 20), 0, 0, 360, cv::Scalar(255), cv::FILLED);   // same object, two places
    cv::ellipse(b, cv::Point(230, 160), cv::Size(40, 20), 0, 0, 360, cv::Scalar(255), cv::FILLED);
    Mat ea, eb;
    REQUIRE(getEmbedding(a, ea, cv::Rect(30, 60, 100, 60), net) == 0);
    REQUIRE(getEmbedding(b, eb, cv::Rect(180, 130, 100, 60), net) == 0);
    CHECK(ea.total() == 50);
    CHECK(cv::norm(ea, eb, cv::NORM_INF) < 1e-4);  // identical crops -> identical embeddings

    Mat bar = Mat::zeros(240, 320, CV_8UC1);
    bar(cv::Rect(100, 100, 120, 12)).setTo(255);
    Mat ebar;
    REQUIRE(getEmbedding(bar, ebar, cv::Rect(90, 90, 140, 40), net) == 0);
    CHECK(cv::norm(ea, ebar, cv::NORM_INF) > 0.05);  // a different shape gives a different embedding

    const auto v = regionEmbedding(a, cv::RotatedRect(cv::Point2f(80, 90), cv::Size2f(80, 40), 0), net);
    CHECK(v.size() == 50);

    Mat out;
    CHECK(getEmbedding(a, out, cv::Rect(500, 500, 10, 10), net) == -1);  // box outside the picture
    CHECK(getEmbedding(Mat(), out, cv::Rect(0, 0, 4, 4), net) == -1);
}
