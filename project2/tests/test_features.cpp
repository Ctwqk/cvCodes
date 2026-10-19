#include <numeric>

#include <opencv2/imgproc.hpp>

#include "doctest.h"
#include "imgfeatures.h"

using cv::Mat;
using cv::Vec3b;

namespace {
Mat solid(int rows, int cols, Vec3b c) { return Mat(rows, cols, CV_8UC3, cv::Scalar(c[0], c[1], c[2])); }
Mat noise(int rows, int cols, unsigned seed) {
    Mat m(rows, cols, CV_8UC3);
    cv::RNG rng(seed);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}
float total(const std::vector<float>& v) { return std::accumulate(v.begin(), v.end(), 0.f); }
// index of the chromaticity bin for (g, r) fractions
int bin(float g, float r) {
    return static_cast<int>(g * 31 + 0.5f) * kHistBins + static_cast<int>(r * 31 + 0.5f);
}
}  // namespace

TEST_CASE("central7x7 returns the 7x7 centre or fails on small images") {
    Mat img = solid(21, 31, Vec3b(1, 2, 3));
    img.at<Vec3b>(10, 15) = Vec3b(9, 8, 7);  // exact centre pixel
    Mat patch;
    REQUIRE(central7x7(img, patch) == 0);
    CHECK(patch.size() == cv::Size(7, 7));
    CHECK(patch.type() == CV_8UC3);
    CHECK(patch.at<Vec3b>(3, 3) == Vec3b(9, 8, 7));
    CHECK(mat2vec(patch).size() == 147);
    Mat small = solid(6, 20, Vec3b(0, 0, 0));
    CHECK(central7x7(small, patch) == -1);
    CHECK(central7x7(Mat(), patch) == -1);
    Mat gray(20, 20, CV_8UC1, cv::Scalar(3));
    CHECK(central7x7(gray, patch) == -1);
}

TEST_CASE("mat2vec flattens row-major and does not modify its input") {
    Mat m = (cv::Mat_<uchar>(2, 3) << 1, 2, 3, 4, 5, 6);
    const std::vector<float> v = mat2vec(m);
    CHECK(v == std::vector<float>{1, 2, 3, 4, 5, 6});
    CHECK(m.type() == CV_8UC1);
    Mat roi = noise(10, 10, 1)(cv::Rect(2, 2, 3, 3));  // non-continuous input
    CHECK(mat2vec(roi).size() == 27);
}

TEST_CASE("histogram: mass equals pixel count, bins follow the g/r chromaticity") {
    Mat img = solid(10, 10, Vec3b(0, 0, 255));  // pure red: g = 0, r = 1
    std::vector<float> h;
    REQUIRE(histogram(img, h) == 0);
    REQUIRE(h.size() == static_cast<size_t>(kHistSize));
    CHECK(total(h) == 100.f);
    CHECK(h[bin(0.f, 1.f)] == 100.f);

    img.setTo(cv::Scalar(0, 255, 0));  // pure green
    REQUIRE(histogram(img, h) == 0);
    CHECK(h[bin(1.f, 0.f)] == 100.f);

    img.setTo(cv::Scalar(255, 0, 0));  // pure blue: g = r = 0 -> bin 0
    REQUIRE(histogram(img, h) == 0);
    CHECK(h[0] == 100.f);

    img.setTo(cv::Scalar(0, 0, 0));  // black pixels must not divide by zero
    REQUIRE(histogram(img, h) == 0);
    CHECK(h[0] == 100.f);
}

TEST_CASE("histogram is brightness invariant and leaves the input untouched") {
    const Mat dim = solid(8, 8, Vec3b(10, 20, 30));
    const Mat bright = solid(8, 8, Vec3b(50, 100, 150));  // same chromaticity
    std::vector<float> a, b;
    Mat copy = dim.clone();
    REQUIRE(histogram(dim, a) == 0);
    REQUIRE(histogram(bright, b) == 0);
    CHECK(a == b);
    CHECK(cv::norm(dim, copy, cv::NORM_INF) == 0);
    CHECK(dim.type() == CV_8UC3);  // the original converted the caller's image to float
    CHECK(histogram(Mat(), a) == -1);
}

TEST_CASE("multiHist keeps the spatial layout the pooled histogram loses") {
    Mat a = solid(60, 80, Vec3b(255, 0, 0));
    Mat b = a.clone();
    a(cv::Rect(30, 20, 21, 21)).setTo(cv::Scalar(0, 0, 255));  // red patch in the centre ROI
    b(cv::Rect(0, 0, 21, 21)).setTo(cv::Scalar(0, 0, 255));    // red patch in the top-left ROI
    std::vector<float> fa, fb, ha, hb;
    REQUIRE(multiHist(a, fa) == 0);
    REQUIRE(multiHist(b, fb) == 0);
    REQUIRE(fa.size() == static_cast<size_t>(kMultiHistRois * kHistSize));
    CHECK(total(fa) == doctest::Approx(kMultiHistRois * 441.0));
    CHECK(fa != fb);  // different ROI got the red patch
    // pooled over all ROIs both images have the same colours
    std::vector<float> pa(kHistSize, 0.f), pb(kHistSize, 0.f);
    for (int r = 0; r < kMultiHistRois; r++)
        for (int i = 0; i < kHistSize; i++) {
            pa[i] += fa[r * kHistSize + i];
            pb[i] += fb[r * kHistSize + i];
        }
    CHECK(pa == pb);
    CHECK(multiHist(solid(10, 10, Vec3b(0, 0, 0)), fa) == -1);  // smaller than one ROI
}

TEST_CASE("texture: 800 bins, every channel of every pixel counted exactly once") {
    const Mat img = noise(40, 50, 3);
    std::vector<float> t;
    REQUIRE(texture(img, t) == 0);
    REQUIRE(t.size() == static_cast<size_t>(kTextureSize));
    CHECK(total(t) == doctest::Approx(3.0 * 40 * 50));
}

TEST_CASE("texture: flat image has no gradient (all mass in one cell) and does not divide by zero") {
    std::vector<float> t;
    REQUIRE(texture(solid(16, 16, Vec3b(40, 40, 40)), t) == 0);
    CHECK(total(t) == doctest::Approx(3.0 * 256));
    int nonZero = 0;
    for (float v : t) nonZero += v > 0;
    CHECK(nonZero == 1);
}

TEST_CASE("texture regression: magnitudes are normalised with (m - min) / (max - min)") {
    // The original used m / (max - min); with min = 5, max = 10 that maps max to bin 62,
    // and with angle 360 (last row) the index lands outside the 800-element vector.
    CHECK(textureBinIndex(0.f, 10.f, 5.f, 10.f) == 31);
    CHECK(textureBinIndex(0.f, 5.f, 5.f, 10.f) == 0);
    CHECK(textureBinIndex(360.f, 10.f, 5.f, 10.f) == 24 * kHistBins + 31);
    CHECK(textureBinIndex(360.f, 10.f, 5.f, 10.f) < kTextureSize);
    CHECK(textureBinIndex(180.f, 7.5f, 5.f, 10.f) == 12 * kHistBins + 16);
    CHECK(textureBinIndex(45.f, 3.f, 3.f, 3.f) == 3 * kHistBins);  // flat image: lowest magnitude bin
    // out-of-range inputs are clamped rather than written out of bounds
    CHECK(textureBinIndex(-5.f, 99.f, 0.f, 10.f) == 31);
    CHECK(textureBinIndex(720.f, -9.f, 0.f, 10.f) == 24 * kHistBins);
}

TEST_CASE("gabor: 33 bins, one count per pixel, flat images are handled") {
    std::vector<float> g;
    REQUIRE(gaborHist(noise(40, 40, 4), g) == 0);
    REQUIRE(g.size() == static_cast<size_t>(kGaborSize));
    CHECK(total(g) == 1600.f);
    REQUIRE(gaborHist(solid(20, 20, Vec3b(77, 77, 77)), g) == 0);
    CHECK(total(g) == 400.f);
    for (float v : g) CHECK(std::isfinite(v));
}

TEST_CASE("fourierHist: 256 values in [0, 11.84], DC block dominates, deterministic") {
    const Mat img = noise(64, 80, 6);
    std::vector<float> f, f2;
    REQUIRE(fourierHist(img, f) == 0);
    REQUIRE(fourierHist(img, f2) == 0);
    REQUIRE(f.size() == static_cast<size_t>(kFourierSize));
    CHECK(f == f2);
    float lo = 1e9f, hi = -1e9f;
    for (float v : f) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    CHECK(lo >= 0.f);
    CHECK(hi <= kDnnMaxValue + 1e-4f);
    CHECK(f[0] == doctest::Approx(kDnnMaxValue));  // DC term is the maximum after normalisation
    CHECK(fourierHist(solid(10, 10, Vec3b(1, 1, 1)), f) == -1);
}

TEST_CASE("fourierHist is blind to a constant brightness offset in the AC part but sees structure") {
    Mat stripes(64, 64, CV_8UC3);
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++) stripes.at<Vec3b>(i, j) = (j / 4) % 2 ? Vec3b(200, 200, 200) : Vec3b(20, 20, 20);
    std::vector<float> fs, fn;
    REQUIRE(fourierHist(stripes, fs) == 0);
    REQUIRE(fourierHist(noise(64, 64, 8), fn) == 0);
    CHECK(fs != fn);
}
