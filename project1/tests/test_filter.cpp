#include <iostream>
#include <set>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "FaceDetect/faceDetect.h"
#include "doctest.h"
#include "filter.h"
#include "filterbank.h"

using cv::Mat;
using cv::Vec3b;

namespace {
Mat randomBGR(int rows, int cols, unsigned seed = 1) {
    Mat m(rows, cols, CV_8UC3);
    cv::RNG rng(seed);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}
bool equalMats(const Mat& a, const Mat& b) {
    return a.size() == b.size() && a.type() == b.type() && cv::norm(a, b, cv::NORM_INF) == 0;
}
// Largest per-element difference on the interior (border excluded).
double interiorDiff(const Mat& a, const Mat& b, int border) {
    const cv::Rect r(border, border, a.cols - 2 * border, a.rows - 2 * border);
    return cv::norm(a(r), b(r), cv::NORM_INF);
}
}  // namespace

TEST_CASE("filters reject empty or wrongly typed input and leave dst alone") {
    Mat gray(4, 4, CV_8UC1, cv::Scalar(5)), dst(2, 2, CV_8UC3, cv::Scalar(7, 7, 7)), empty;
    CHECK(myRGB2Gray(gray, dst) == -1);
    CHECK(sepia(empty, dst) == -1);
    CHECK(blur5x5_1(gray, dst) == -1);
    CHECK(blur5x5_2(gray, dst) == -1);
    CHECK(sobelX3x3(gray, dst) == -1);
    CHECK(sobelY3x3(gray, dst) == -1);
    CHECK(blurQuantize(gray, dst) == -1);
    CHECK(confuseRedGreen(gray, dst) == -1);
    CHECK(localMax(empty, dst) == -1);
    CHECK(dst.size() == cv::Size(2, 2));
    CHECK(dst.at<Vec3b>(0, 0) == Vec3b(7, 7, 7));
}

TEST_CASE("myRGB2Gray is 255 minus the channel average (regression: it used to return the blue channel)") {
    Mat img(1, 3, CV_8UC3);
    img.at<Vec3b>(0, 0) = Vec3b(30, 60, 90);   // average 60  -> 195
    img.at<Vec3b>(0, 1) = Vec3b(255, 255, 255);  // white -> 0
    img.at<Vec3b>(0, 2) = Vec3b(0, 0, 0);        // black -> 255
    Mat g;
    REQUIRE(myRGB2Gray(img, g) == 0);
    REQUIRE(g.type() == CV_8UC1);
    CHECK(g.at<uchar>(0, 0) == 195);
    CHECK(g.at<uchar>(0, 1) == 0);
    CHECK(g.at<uchar>(0, 2) == 255);
}

TEST_CASE("every filter may write into its own input") {
    const Mat img = randomBGR(40, 50);
    Mat expected, inplace = img.clone();
    REQUIRE(sepia(img, expected) == 0);
    REQUIRE(sepia(inplace, inplace) == 0);
    CHECK(equalMats(expected, inplace));
    inplace = img.clone();
    REQUIRE(blur5x5_2(img, expected) == 0);
    REQUIRE(blur5x5_2(inplace, inplace) == 0);
    CHECK(equalMats(expected, inplace));
    inplace = img.clone();
    REQUIRE(blurQuantize(img, expected) == 0);
    REQUIRE(blurQuantize(inplace, inplace) == 0);
    CHECK(equalMats(expected, inplace));
}

TEST_CASE("sepia: neutral grey keeps the matrix row sums, corners are vignetted to black, no stdout noise") {
    Mat img(41, 41, CV_8UC3, cv::Scalar(100, 100, 100));
    std::ostringstream captured;
    std::streambuf* old = std::cout.rdbuf(captured.rdbuf());
    Mat out;
    const int rc = sepia(img, out);
    std::cout.rdbuf(old);
    REQUIRE(rc == 0);
    CHECK(captured.str().empty());  // the original printed "her1"/"hrer" on every call
    const Vec3b centre = out.at<Vec3b>(20, 20);
    // row sums of the sepia matrix: 0.937, 1.203, 1.351 (blue, green, red outputs)
    CHECK(centre[0] == 94);
    CHECK(centre[1] == 120);
    CHECK(centre[2] == 135);
    CHECK(out.at<Vec3b>(0, 0) == Vec3b(0, 0, 0));
    CHECK(out.at<Vec3b>(20, 20)[2] > out.at<Vec3b>(20, 3)[2]);  // darker towards the edge
}

TEST_CASE("blurs: constant image is unchanged (the old accumulation lost grey levels)") {
    Mat img(30, 30, CV_8UC3, cv::Scalar(200, 150, 99));
    Mat a, b;
    REQUIRE(blur5x5_1(img, a) == 0);
    REQUIRE(blur5x5_2(img, b) == 0);
    CHECK(equalMats(a, img));
    CHECK(equalMats(b, img));
}

TEST_CASE("blurs: impulse response is the 5x5 kernel / 100") {
    Mat img = Mat::zeros(15, 15, CV_8UC3);
    img.at<Vec3b>(7, 7) = Vec3b(255, 255, 255);
    Mat a, b;
    REQUIRE(blur5x5_1(img, a) == 0);
    REQUIRE(blur5x5_2(img, b) == 0);
    const int k[5] = {1, 2, 4, 2, 1};
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            const int expected = static_cast<int>(255.0 * k[i + 2] * k[j + 2] / 100.0 + 0.5);
            CHECK(a.at<Vec3b>(7 + i, 7 + j)[1] == expected);
            CHECK(b.at<Vec3b>(7 + i, 7 + j)[1] == expected);
        }
    }
    CHECK(a.at<Vec3b>(7, 10) == Vec3b(0, 0, 0));
}

TEST_CASE("blur5x5_1 and blur5x5_2 agree exactly away from the 2-pixel border; border is copied") {
    const Mat img = randomBGR(37, 53, 7);
    Mat a, b;
    REQUIRE(blur5x5_1(img, a) == 0);
    REQUIRE(blur5x5_2(img, b) == 0);
    CHECK(equalMats(a, b));
    const cv::Rect inner(2, 2, img.cols - 4, img.rows - 4);
    Mat borderA = a.clone(), borderSrc = img.clone();
    borderA(inner).setTo(0);
    borderSrc(inner).setTo(0);
    CHECK(equalMats(borderA, borderSrc));
    // and it really blurs: compare against OpenCV's Gaussian within rounding
    Mat ref;
    cv::GaussianBlur(img, ref, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
    // GaussianBlur's default 5x5 kernel is [1 4 6 4 1]/16, a different kernel, so
    // only check that both reduce the variance similarly.
    cv::Scalar m1, s1, m2, s2;
    cv::meanStdDev(img, m1, s1);
    cv::meanStdDev(a, m2, s2);
    CHECK(s2[0] < s1[0]);
}

TEST_CASE("blurs: images smaller than the kernel are returned unchanged") {
    const Mat tiny = randomBGR(4, 4);
    Mat a, b;
    REQUIRE(blur5x5_1(tiny, a) == 0);
    REQUIRE(blur5x5_2(tiny, b) == 0);
    CHECK(equalMats(a, tiny));
    CHECK(equalMats(b, tiny));
}

TEST_CASE("Sobel: constant image gives zeros everywhere, including the border") {
    Mat img(20, 25, CV_8UC3, cv::Scalar(10, 200, 77));
    Mat sx, sy;
    REQUIRE(sobelX3x3(img, sx) == 0);
    REQUIRE(sobelY3x3(img, sy) == 0);
    CHECK(sx.type() == CV_16SC3);
    CHECK(cv::countNonZero(sx.reshape(1)) == 0);  // used to contain raw pixel values on the border
    CHECK(cv::countNonZero(sy.reshape(1)) == 0);
}

TEST_CASE("Sobel: vertical step edge") {
    Mat img = Mat::zeros(10, 10, CV_8UC3);
    img(cv::Rect(5, 0, 5, 10)).setTo(cv::Scalar(255, 255, 255));  // dark left, bright right
    Mat sx, sy;
    REQUIRE(sobelX3x3(img, sx) == 0);
    REQUIRE(sobelY3x3(img, sy) == 0);
    CHECK(sx.at<cv::Vec3s>(5, 4)[0] == 1020);  // 4 * 255, positive: intensity rises to the right
    CHECK(sx.at<cv::Vec3s>(5, 5)[0] == 1020);
    CHECK(sx.at<cv::Vec3s>(5, 2)[0] == 0);
    CHECK(sx.at<cv::Vec3s>(5, 8)[0] == 0);
    // column smoothing of the step only affects rows near the top/bottom border, which stay zero
    CHECK(sy.at<cv::Vec3s>(5, 5)[0] == 0);
}

TEST_CASE("Sobel matches cv::Sobel on the interior (Y has the opposite, course, sign)") {
    const Mat img = randomBGR(30, 40, 3);
    Mat sx, sy, rx, ry;
    REQUIRE(sobelX3x3(img, sx) == 0);
    REQUIRE(sobelY3x3(img, sy) == 0);
    cv::Sobel(img, rx, CV_16S, 1, 0, 3);
    cv::Sobel(img, ry, CV_16S, 0, 1, 3);
    CHECK(interiorDiff(sx, rx, 1) == 0);
    CHECK(interiorDiff(sy, -ry, 1) == 0);
}

TEST_CASE("magnitude: 3-4-5, saturation, and input validation") {
    Mat sx(1, 2, CV_16SC3, cv::Scalar(3, 3, 3)), sy(1, 2, CV_16SC3, cv::Scalar(4, 4, 4));
    Mat out;
    REQUIRE(magnitude(sx, sy, out) == 0);
    CHECK(out.type() == CV_8UC3);
    CHECK(out.at<Vec3b>(0, 0) == Vec3b(5, 5, 5));
    sx.setTo(cv::Scalar(1020, -1020, 0));
    sy.setTo(cv::Scalar(1020, 1020, 300));
    REQUIRE(magnitude(sx, sy, out) == 0);
    CHECK(out.at<Vec3b>(0, 0) == Vec3b(255, 255, 255));  // 1442 used to wrap around to a small value
    CHECK(out.at<Vec3b>(0, 1)[2] == 255);
    Mat other(2, 2, CV_16SC3);
    CHECK(magnitude(sx, other, out) == -1);
    CHECK(magnitude(Mat(1, 2, CV_8UC3), sy, out) == -1);
}

TEST_CASE("blurQuantize: only multiples of the bucket size, levels validated") {
    const Mat img = randomBGR(30, 30, 5);
    Mat q;
    REQUIRE(blurQuantize(img, q, 10) == 0);  // bucket 255 / 10 = 25
    for (int i = 0; i < q.rows; i++)
        for (int j = 0; j < q.cols; j++)
            for (int c = 0; c < 3; c++) CHECK(q.at<Vec3b>(i, j)[c] % 25 == 0);
    Mat flat(20, 20, CV_8UC3, cv::Scalar(200, 200, 200));
    REQUIRE(blurQuantize(flat, q, 10) == 0);
    CHECK(q.at<Vec3b>(10, 10) == Vec3b(200, 200, 200));
    Mat dst;
    CHECK(blurQuantize(img, dst, 0) == -1);
    CHECK(blurQuantize(img, dst, 256) == -1);
    Mat full, blurred;
    REQUIRE(blurQuantize(img, full, 255) == 0);  // bucket 1: identical to the plain blur
    REQUIRE(blur5x5_2(img, blurred) == 0);
    CHECK(equalMats(full, blurred));
}

TEST_CASE("confuseRedGreen") {
    Mat img(1, 1, CV_8UC3, cv::Scalar(10, 100, 200));  // B, G, R
    Mat out;
    REQUIRE(confuseRedGreen(img, out) == 0);
    CHECK(out.at<Vec3b>(0, 0) == Vec3b(150, 10, 10));
}

TEST_CASE("localMax equals an 11x11 dilation on the interior and copies the 5-pixel frame") {
    const Mat img = randomBGR(40, 45, 9);
    Mat out, ref;
    REQUIRE(localMax(img, out) == 0);
    cv::dilate(img, ref, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
    CHECK(interiorDiff(out, ref, 5) == 0);
    Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    Mat outGray;
    REQUIRE(localMax(gray, outGray) == 0);  // works on single-channel images too
    CHECK(outGray.channels() == 1);
    CHECK(outGray.at<uchar>(2, 2) == gray.at<uchar>(2, 2));  // frame copied
    const Mat small = randomBGR(8, 8);
    REQUIRE(localMax(small, out) == 0);
    CHECK(equalMats(out, small));
}

TEST_CASE("brightAndContrast saturates") {
    Mat img(1, 2, CV_8UC3);
    img.at<Vec3b>(0, 0) = Vec3b(200, 100, 10);
    img.at<Vec3b>(0, 1) = Vec3b(0, 255, 128);
    Mat out;
    REQUIRE(brightAndContrast(img, out, 2.0f, 0) == 0);
    CHECK(out.at<Vec3b>(0, 0) == Vec3b(255, 200, 20));
    REQUIRE(brightAndContrast(img, out, 1.0f, -50) == 0);
    CHECK(out.at<Vec3b>(0, 0) == Vec3b(150, 50, 0));
    CHECK(out.at<Vec3b>(0, 1) == Vec3b(0, 205, 78));
}

TEST_CASE("kaleidoscope: every tile is mirror symmetric; invalid arguments are rejected") {
    const int num = 3, size = 8, side = num * size;
    const Mat img = randomBGR(side, side, 2);
    const Mat mask = makeKaleidoscopeMask(size);
    Mat out;
    REQUIRE(kaleidoscope(img, out, mask, num, size) == 0);
    REQUIRE(out.size() == img.size());
    for (int ti = 0; ti < num; ti++)
        for (int tj = 0; tj < num; tj++)
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++) {
                    const Vec3b p = out.at<Vec3b>(tj * size + i, ti * size + j);
                    CHECK(p == out.at<Vec3b>(tj * size + size - 1 - i, ti * size + j));
                    CHECK(p == out.at<Vec3b>(tj * size + i, ti * size + size - 1 - j));
                }
    // the top-left quadrant of each tile is preserved
    CHECK(out.at<Vec3b>(1, 1) == img.at<Vec3b>(1, 1));
    CHECK(kaleidoscope(randomBGR(10, 12), out, mask, num, size) == -1);  // not square
    CHECK(kaleidoscope(img, out, mask, num + 1, size) == -1);            // tiles exceed the image
    CHECK(kaleidoscope(img, out, makeKaleidoscopeMask(size + 1), num, size) == -1);  // wrong mask size
}

TEST_CASE("centralize zooms to 640x480 and stays inside the frame") {
    Mat frame = randomBGR(480, 640);
    cv::Rect face(300, 200, 60, 70);
    REQUIRE(centralize(frame, face) == 0);
    CHECK(frame.size() == cv::Size(640, 480));

    Mat edge = randomBGR(480, 640);
    cv::Rect corner(600, 440, 40, 40);  // touches the bottom-right corner
    REQUIRE(centralize(edge, corner) == 0);
    CHECK(edge.size() == cv::Size(640, 480));

    Mat big = randomBGR(100, 100);
    cv::Rect huge(0, 0, 60, 60);  // already larger than half the frame: no zoom
    REQUIRE(centralize(big, huge) == 0);
    CHECK(big.size() == cv::Size(100, 100));
    Mat none;
    cv::Rect r(0, 0, 1, 1);
    CHECK(centralize(none, r) == -1);
}

TEST_CASE("filter bank: unique names and keys, every filter runs on a test picture") {
    std::set<std::string> names;
    std::set<char> keys;
    for (const FilterEntry& e : filterBank()) {
        CHECK(names.insert(e.name).second);
        CHECK(keys.insert(e.key).second);
        CHECK(findFilterByName(e.name) == &e);
        CHECK(findFilterByKey(e.key) == &e);
    }
    CHECK(findFilterByName("nope") == nullptr);
    CHECK(findFilterByKey('#') == nullptr);

    const Mat img = randomBGR(96, 128, 4);
    const bool haveCascade = !findFaceCascade().empty();
    CHECK(haveCascade);
    for (const FilterEntry& e : filterBank()) {
        Mat out;
        INFO("filter " << e.name);
        REQUIRE(e.apply(img, out) == 0);
        CHECK(out.depth() == CV_8U);
        CHECK_FALSE(out.empty());
    }
}

TEST_CASE("face detection on a blank picture finds nothing") {
    Mat grey(120, 160, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> faces{cv::Rect(1, 2, 3, 4)};
    REQUIRE(detectFaces(grey, faces) == 0);
    CHECK(faces.empty());
    Mat frame(120, 160, CV_8UC3, cv::Scalar(0));
    std::vector<cv::Rect> boxes{cv::Rect(10, 10, 80, 80), cv::Rect(0, 0, 20, 20)};
    drawBoxes(frame, boxes);  // second box is narrower than minWidth: not drawn
    CHECK(frame.at<Vec3b>(10, 50) != Vec3b(0, 0, 0));
    CHECK(frame.at<Vec3b>(0, 10) == Vec3b(0, 0, 0));
}
