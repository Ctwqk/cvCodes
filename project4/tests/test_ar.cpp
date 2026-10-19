#include <cstdio>
#include <filesystem>
#include <fstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "ar_utils.h"
#include "doctest.h"

using namespace p4;
using cv::Mat;

namespace {
Intrinsics syntheticCamera() {
    Intrinsics cam;
    cam.cameraMatrix = (cv::Mat_<double>(3, 3) << 520, 0, 320, 0, 520, 240, 0, 0, 1);
    cam.distCoeffs = {-0.18, 0.05, 0.0005, -0.0003, 0.0};
    return cam;
}
const BoardTemplate& board() {
    static BoardTemplate b = [] {
        BoardTemplate t;
        std::string err;
        const bool ok = loadBoardTemplate(cvc::repoPath("project4/checkerboard.png"), cv::Size(9, 6), t, &err);
        REQUIRE_MESSAGE(ok, err);
        return t;
    }();
    return b;
}
}  // namespace

TEST_CASE("chessboardPoints follows OpenCV's corner order: x right, y up, z = 0") {
    const auto pts = chessboardPoints(cv::Size(9, 6));
    REQUIRE(pts.size() == 54);
    CHECK(pts[0] == cv::Point3f(0, 0, 0));
    CHECK(pts[8] == cv::Point3f(8, 0, 0));
    CHECK(pts[9] == cv::Point3f(0, -1, 0));
    CHECK(pts[53] == cv::Point3f(8, -5, 0));
    CHECK(chessboardPoints(cv::Size(3, 2)).size() == 6);
}

TEST_CASE("intrinsics files round-trip and the original file parses") {
    Intrinsics orig;
    std::string err;
    REQUIRE_MESSAGE(readIntrinsics(cvc::repoPath("project4/intcPara.txt"), orig, &err), err);
    CHECK(orig.pattern == cv::Size(9, 6));
    CHECK(orig.cameraMatrix.at<double>(0, 0) == doctest::Approx(446.697));
    CHECK(orig.cameraMatrix.at<double>(0, 2) == doctest::Approx(320));
    CHECK(orig.cameraMatrix.at<double>(2, 2) == doctest::Approx(1));
    REQUIRE(orig.distCoeffs.size() == 5);
    CHECK(orig.distCoeffs[0] == doctest::Approx(0.215843));

    const std::string path = (std::filesystem::temp_directory_path() / "cvc_test_intrinsics.txt").string();
    REQUIRE(writeIntrinsics(path, orig, &err));
    Intrinsics back;
    REQUIRE(readIntrinsics(path, back, &err));
    CHECK(back.pattern == orig.pattern);
    CHECK(cv::norm(back.cameraMatrix, orig.cameraMatrix, cv::NORM_INF) < 1e-6);
    REQUIRE(back.distCoeffs.size() == 5);
    CHECK(back.distCoeffs[4] == doctest::Approx(orig.distCoeffs[4]));
    std::remove(path.c_str());
}

TEST_CASE("readIntrinsics rejects missing and truncated files") {
    Intrinsics i;
    std::string err;
    CHECK_FALSE(readIntrinsics("/definitely/not/here.txt", i, &err));
    CHECK_FALSE(err.empty());
    const std::string path = (std::filesystem::temp_directory_path() / "cvc_test_bad_intrinsics.txt").string();
    std::ofstream(path) << "9 6\n1 0 320 0 1\n";
    CHECK_FALSE(readIntrinsics(path, i, &err));
    std::ofstream(path) << "x y\n";
    CHECK_FALSE(readIntrinsics(path, i, &err));
    std::remove(path.c_str());
    Intrinsics unset;
    CHECK_FALSE(writeIntrinsics("/definitely/not/here/out.txt", unset));
}

TEST_CASE("calibration recovers a known camera from exact projections") {
    const Intrinsics truth = syntheticCamera();
    std::vector<std::vector<cv::Point2f>> views;
    for (int i = 0; i < 12; i++) {
        Mat rvec, tvec;
        samplePose(i, 12, 5, rvec, tvec);
        std::vector<cv::Point2f> pts;
        cv::projectPoints(chessboardPoints(truth.pattern), rvec, tvec, truth.cameraMatrix, truth.distCoeffs, pts);
        views.push_back(pts);
    }
    CalibrationResult r;
    REQUIRE(calibrateFromCorners(cv::Size(640, 480), truth.pattern, views, r));
    CHECK(r.views == 12);
    CHECK(r.rms < 0.05);
    CHECK(r.cameraMatrix.at<double>(0, 0) == doctest::Approx(520).epsilon(0.01));
    CHECK(r.cameraMatrix.at<double>(1, 1) == doctest::Approx(520).epsilon(0.01));  // aspect ratio fixed to 1
    CHECK(r.cameraMatrix.at<double>(0, 2) == doctest::Approx(320).epsilon(0.02));
    CHECK(r.cameraMatrix.at<double>(1, 2) == doctest::Approx(240).epsilon(0.02));
    REQUIRE(r.distCoeffs.size() >= 2);
    CHECK(r.distCoeffs[0] == doctest::Approx(-0.18).epsilon(0.05));
}

TEST_CASE("calibration needs at least three views") {
    CalibrationResult r;
    std::vector<std::vector<cv::Point2f>> two(2, std::vector<cv::Point2f>(54));
    CHECK_FALSE(calibrateFromCorners(cv::Size(640, 480), cv::Size(9, 6), two, r));
}

TEST_CASE("synthetic poses keep the board inside the frame and are reproducible") {
    const Intrinsics cam = syntheticCamera();
    for (int i = 0; i < 40; i++) {
        Mat rvec, tvec, rvec2, tvec2;
        samplePose(i, 40, 3, rvec, tvec);
        samplePose(i, 40, 3, rvec2, tvec2);
        CHECK(cv::norm(rvec, rvec2) == 0);
        CHECK(cv::norm(tvec, tvec2) == 0);
        std::vector<cv::Point3f> outline = {{-1, 1, 0}, {9, 1, 0}, {9, -6, 0}, {-1, -6, 0}};  // the printed board
        std::vector<cv::Point2f> px;
        cv::projectPoints(outline, rvec, tvec, cam.cameraMatrix, cam.distCoeffs, px);
        for (const auto& p : px) {
            CHECK(p.x > 0);
            CHECK(p.x < 640);
            CHECK(p.y > 0);
            CHECK(p.y < 480);
        }
    }
    Mat a, b, c, d;
    samplePose(0, 2, 1, a, b);
    samplePose(1, 2, 1, c, d);
    CHECK(cv::norm(a, c) > 0);  // different views
}

TEST_CASE("rendered views contain a detectable board whose pose matches the one used to draw it") {
    const Intrinsics cam = syntheticCamera();
    for (int i = 0; i < 4; i++) {
        Mat rvec, tvec;
        samplePose(i, 4, 11, rvec, tvec);
        const Mat view = renderBoardView(board(), cam, rvec, tvec, cv::Size(640, 480), 7);
        REQUIRE(view.size() == cv::Size(640, 480));
        REQUIRE(view.type() == CV_8UC3);
        Mat gray;
        cv::cvtColor(view, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        INFO("view " << i);
        REQUIRE(findBoardCorners(gray, cam.pattern, corners));
        REQUIRE(corners.size() == 54);
        // detected corners agree with the projection of the true board points
        std::vector<cv::Point2f> expected;
        cv::projectPoints(chessboardPoints(cam.pattern), rvec, tvec, cam.cameraMatrix, cam.distCoeffs, expected);
        double worst = 0;
        for (size_t k = 0; k < corners.size(); k++) worst = std::max(worst, cv::norm(corners[k] - expected[k]));
        CHECK(worst < 1.5);
        // and the recovered pose is close to the true one
        Mat r2, t2;
        REQUIRE(estimatePose(cam, corners, r2, t2));
        CHECK(cv::norm(t2, tvec) < 0.05 * cv::norm(tvec));
        CHECK(reprojectionError(cam, corners, r2, t2) < 1.0);
    }
}

TEST_CASE("end to end: calibrating from rendered views recovers the camera") {
    const Intrinsics truth = syntheticCamera();
    std::vector<std::vector<cv::Point2f>> views;
    for (int i = 0; i < 15; i++) {
        Mat rvec, tvec;
        samplePose(i, 15, 2, rvec, tvec);
        const Mat view = renderBoardView(board(), truth, rvec, tvec, cv::Size(640, 480), 100 + i);
        Mat gray;
        cv::cvtColor(view, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        if (findBoardCorners(gray, truth.pattern, corners)) views.push_back(corners);
    }
    REQUIRE(views.size() >= 12);
    CalibrationResult r;
    REQUIRE(calibrateFromCorners(cv::Size(640, 480), truth.pattern, views, r));
    CHECK(r.rms < 0.6);
    CHECK(r.cameraMatrix.at<double>(0, 0) == doctest::Approx(520).epsilon(0.03));
    CHECK(r.cameraMatrix.at<double>(0, 2) == doctest::Approx(320).epsilon(0.05));
    CHECK(r.cameraMatrix.at<double>(1, 2) == doctest::Approx(240).epsilon(0.05));
}

TEST_CASE("findBoardCorners rejects pictures without a board") {
    Mat blank(240, 320, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Point2f> corners;
    CHECK_FALSE(findBoardCorners(blank, cv::Size(9, 6), corners));
    CHECK(corners.empty());
    CHECK_FALSE(findBoardCorners(Mat(), cv::Size(9, 6), corners));
    CHECK_FALSE(findBoardCorners(Mat(10, 10, CV_8UC3), cv::Size(9, 6), corners));
    CHECK_FALSE(findBoardCorners(Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Size(9, 6), corners));  // used to throw
}

TEST_CASE("harrisCorners finds the four corners of a square, one point each") {
    Mat img(120, 160, CV_8UC1, cv::Scalar(30));
    cv::rectangle(img, cv::Rect(40, 30, 60, 50), cv::Scalar(220), cv::FILLED);
    const auto pts = harrisCorners(img, 100);
    REQUIRE(pts.size() >= 4);
    const std::vector<cv::Point2f> expected = {{40, 30}, {99, 30}, {99, 79}, {40, 79}};
    for (const auto& e : expected) {
        double best = 1e9;
        for (const auto& p : pts) best = std::min(best, cv::norm(p - e));
        CHECK(best <= 3.0);
    }
    for (const auto& p : pts) {  // nothing in the flat interior or the empty background
        CHECK(cv::norm(p - cv::Point2f(70, 55)) > 10);
        CHECK(cv::norm(p - cv::Point2f(10, 10)) > 10);
    }
    CHECK(harrisCorners(img, 100, false).size() >= pts.size());  // without suppression: at least as many
    CHECK(harrisCorners(Mat(120, 160, CV_8UC1, cv::Scalar(77)), 10).empty());  // flat image
    CHECK(harrisCorners(Mat(), 10).empty());
    CHECK(harrisCorners(img, 255.5).empty());  // above the maximum possible response
}

TEST_CASE("replaceBoard covers the whole printed board but nothing else") {
    const Intrinsics cam = syntheticCamera();
    Mat rvec, tvec;
    samplePose(2, 4, 11, rvec, tvec);
    Mat frame = renderBoardView(board(), cam, rvec, tvec, cv::Size(640, 480), 3);
    const Mat before = frame.clone();
    Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    REQUIRE(findBoardCorners(gray, cam.pattern, corners));

    Mat picture(100, 150, CV_8UC3, cv::Scalar(0, 200, 0));  // solid green
    REQUIRE(replaceBoard(frame, picture, corners, cam.pattern, &gray));
    // the middle of the board is now green
    std::vector<cv::Point2f> centre;
    cv::projectPoints(std::vector<cv::Point3f>{{4, -2.5f, 0}}, rvec, tvec, cam.cameraMatrix, cam.distCoeffs, centre);
    const cv::Vec3b mid = frame.at<cv::Vec3b>(cvRound(centre[0].y), cvRound(centre[0].x));
    CHECK(mid[1] > 150);
    CHECK(mid[0] < 60);
    // a pixel in the table, far from the board, is unchanged
    CHECK(frame.at<cv::Vec3b>(2, 2) == before.at<cv::Vec3b>(2, 2));
    // the grey search image has the board blacked out, so it is not found a second time
    std::vector<cv::Point2f> again;
    CHECK_FALSE(findBoardCorners(gray, cam.pattern, again));
    // wrong number of corners
    corners.pop_back();
    CHECK_FALSE(replaceBoard(frame, picture, corners, cam.pattern));
}

TEST_CASE("AR drawing changes the frame, and the pose read-out does not crash on float vectors") {
    const Intrinsics cam = syntheticCamera();
    Mat rvec, tvec;
    samplePose(0, 4, 11, rvec, tvec);
    Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    drawAxes(frame, cam, rvec, tvec);
    const int axes = cv::countNonZero(frame.reshape(1));
    CHECK(axes > 0);
    drawStar(frame, cam, rvec, tvec);
    CHECK(cv::countNonZero(frame.reshape(1)) > axes);
    Mat rf, tf;
    rvec.convertTo(rf, CV_32F);
    tvec.convertTo(tf, CV_32F);
    drawPoseText(frame, rf, tf);
}

namespace {
// angle (radians) of the rotation taking pose a to pose b (rotation vectors near 180 degrees are not
// continuous, so compare rotation matrices)
double rotationAngle(const Mat& ra, const Mat& rb) {
    Mat A, B;
    cv::Rodrigues(ra, A);
    cv::Rodrigues(rb, B);
    const Mat D = A.t() * B;
    return std::acos(std::max(-1.0, std::min(1.0, (cv::trace(D)[0] - 1) / 2)));
}
}  // namespace

TEST_CASE("trajectory poses vary smoothly") {
    Mat r0, t0, r1, t1;
    trajectoryPose(0.0, r0, t0);
    trajectoryPose(0.01, r1, t1);
    CHECK(cv::norm(t0, t1) < 0.5);
    CHECK(rotationAngle(r0, r1) < 0.1);
    Mat r2, t2;
    trajectoryPose(0.4, r2, t2);
    CHECK(rotationAngle(r0, r2) > 0.2);
}
