#include <map>
#include <random>
#include <set>

#include "colorquant.h"
#include "doctest.h"
#include "kmeans.h"

using cv::Vec3b;

namespace {
std::vector<Vec3b> twoBlobs() {
    std::vector<Vec3b> d;
    for (int i = 0; i < 20; ++i) d.push_back(Vec3b(10 + i % 3, 10, 10));      // dark blue-ish
    for (int i = 0; i < 20; ++i) d.push_back(Vec3b(200, 200 + i % 3, 200));   // light
    return d;
}
}  // namespace

TEST_CASE("kmeans separates two well separated blobs") {
    auto data = twoBlobs();
    std::vector<Vec3b> means;
    std::vector<int> labels(data.size());
    REQUIRE(kmeans(data, means, labels.data(), 2, 20) == 0);
    REQUIRE(means.size() == 2);
    const int lo = means[0][0] < means[1][0] ? 0 : 1;
    CHECK(means[lo][0] <= 12);
    CHECK(means[1 - lo][0] == 200);
    for (size_t i = 0; i < data.size(); ++i) {
        const bool isDark = data[i][0] < 100;
        CHECK((labels[i] == lo) == isDark);
    }
}

TEST_CASE("K = 1 gives the (rounded) mean color") {
    std::vector<Vec3b> data = {Vec3b(0, 0, 0), Vec3b(10, 20, 31)};
    std::vector<Vec3b> means;
    std::vector<int> labels(2);
    REQUIRE(kmeans(data, means, labels.data(), 1) == 0);
    CHECK(means[0] == Vec3b(5, 10, 16));  // 15.5 rounds up
}

TEST_CASE("regression: data.size() a multiple of K used to divide by zero") {
    for (int n : {6, 12, 40}) {
        for (int K : {1, 2, 3, 6}) {
            if (n % K != 0) continue;
            std::vector<Vec3b> data;
            for (int i = 0; i < n; ++i) data.push_back(Vec3b(static_cast<uchar>(i * 5), 0, 0));
            std::vector<Vec3b> means;
            std::vector<int> labels(data.size());
            CHECK(kmeans(data, means, labels.data(), K) == 0);
            CHECK(means.size() == static_cast<size_t>(K));
        }
    }
}

TEST_CASE("invalid arguments are rejected") {
    std::vector<Vec3b> data(3, Vec3b(1, 2, 3));
    std::vector<Vec3b> means;
    std::vector<int> labels(3);
    CHECK(kmeans(data, means, labels.data(), 4) == -1);   // K > N
    CHECK(kmeans(data, means, labels.data(), 0) == -1);   // K < 1
    CHECK(kmeans({}, means, labels.data(), 1) == -1);     // no data
    CHECK(kmeans(data, means, nullptr, 1) == -1);
}

TEST_CASE("empty clusters keep their previous mean instead of crashing") {
    // Only two distinct colors but K = 4: some clusters end up empty.
    std::vector<Vec3b> data(8, Vec3b(0, 0, 0));
    for (int i = 0; i < 8; ++i) data.push_back(Vec3b(255, 255, 255));
    std::vector<Vec3b> means;
    std::vector<int> labels(data.size());
    REQUIRE(kmeans(data, means, labels.data(), 4, 10) == 0);
    std::set<int> used(labels.begin(), labels.end());
    CHECK(used.size() >= 2);
}

TEST_CASE("labels always match the nearest of the returned means") {
    std::mt19937 rng(5);
    std::vector<Vec3b> data;
    for (int i = 0; i < 500; ++i) {
        data.push_back(Vec3b(static_cast<uchar>(rng() % 256), static_cast<uchar>(rng() % 256),
                             static_cast<uchar>(rng() % 256)));
    }
    std::vector<Vec3b> means;
    std::vector<int> labels(data.size());
    REQUIRE(kmeans(data, means, labels.data(), 7, 3) == 0);  // few iterations: not converged
    for (size_t i = 0; i < data.size(); ++i) {
        const int own = ssd3(data[i], means[static_cast<size_t>(labels[i])]);
        for (const Vec3b& m : means) CHECK(own <= ssd3(data[i], m));
    }
}

TEST_CASE("same seed gives the same result, K-means never increases the error") {
    std::mt19937 rng(11);
    std::vector<Vec3b> data;
    for (int i = 0; i < 300; ++i) {
        data.push_back(Vec3b(static_cast<uchar>(rng() % 256), static_cast<uchar>(rng() % 256),
                             static_cast<uchar>(rng() % 256)));
    }
    std::vector<Vec3b> m1, m2, m3;
    std::vector<int> l1(data.size()), l2(data.size()), l3(data.size());
    REQUIRE(kmeans(data, m1, l1.data(), 5, 15, 0, 3) == 0);
    REQUIRE(kmeans(data, m2, l2.data(), 5, 15, 0, 3) == 0);
    CHECK(m1 == m2);
    CHECK(l1 == l2);
    REQUIRE(kmeans(data, m3, l3.data(), 5, 1, 0, 3) == 0);
    auto sse = [&](const std::vector<Vec3b>& m, const std::vector<int>& l) {
        long long s = 0;
        for (size_t i = 0; i < data.size(); ++i) s += ssd3(data[i], m[static_cast<size_t>(l[i])]);
        return s;
    };
    CHECK(sse(m1, l1) <= sse(m3, l3));
}

TEST_CASE("ssd3 mixes element types without overflow") {
    CHECK(ssd3(Vec3b(255, 0, 0), Vec3b(0, 0, 0)) == 65025);
    CHECK(ssd3(cv::Vec4i(1, 2, 3, 99), Vec3b(1, 2, 5)) == 4);
}

TEST_CASE("sampleColors takes one pixel per full block") {
    cv::Mat img(10, 9, CV_8UC3, cv::Scalar(1, 2, 3));
    std::mt19937 rng(1);
    CHECK(sampleColors(img, 4, rng).size() == 2u * 2u);   // floor(10/4) x floor(9/4)
    CHECK(sampleColors(img, 1, rng).size() == 90u);
    CHECK(sampleColors(img, 20, rng).empty());
}

TEST_CASE("quantize maps every pixel to a palette color and paletteShares sums to 1") {
    cv::Mat img(20, 20, CV_8UC3);
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            img.at<Vec3b>(i, j) = j < 5 ? Vec3b(250, 5, 5) : Vec3b(5, 5, 250);
    const std::vector<Vec3b> palette = {Vec3b(255, 0, 0), Vec3b(0, 0, 255), Vec3b(0, 255, 0)};
    cv::Mat q = quantize(img, palette);
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j) {
            const Vec3b p = q.at<Vec3b>(i, j);
            CHECK((p == palette[0] || p == palette[1] || p == palette[2]));
        }
    const auto shares = paletteShares(img, palette);
    CHECK(shares[0] == doctest::Approx(0.25));
    CHECK(shares[1] == doctest::Approx(0.75));
    CHECK(shares[2] == doctest::Approx(0.0));
    CHECK(meanSquaredError(img, img) == 0.0);
    CHECK(meanSquaredError(img, q) > 0.0);
}

TEST_CASE("more colors never quantize worse (on a gradient)") {
    cv::Mat img(64, 64, CV_8UC3);
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j)
            img.at<Vec3b>(i, j) = Vec3b(static_cast<uchar>(i * 4), static_cast<uchar>(j * 4), 128);
    double prev = 1e18;
    for (int K : {2, 8, 32}) {
        std::mt19937 rng(1);
        auto data = sampleColors(img, 1, rng);
        std::vector<Vec3b> means;
        std::vector<int> labels(data.size());
        REQUIRE(kmeans(data, means, labels.data(), K, 20, 0, 1) == 0);
        const double mse = meanSquaredError(img, quantize(img, means));
        CHECK(mse < prev);
        prev = mse;
    }
}

TEST_CASE("paletteStrip and toHex") {
    const std::vector<Vec3b> palette = {Vec3b(0, 0, 255), Vec3b(255, 0, 0)};  // red, blue (BGR)
    cv::Mat strip = paletteStrip(palette, {0.75, 0.25}, 100, 10);
    CHECK(strip.cols == 100);
    CHECK(strip.at<Vec3b>(5, 10) == palette[0]);
    CHECK(strip.at<Vec3b>(5, 90) == palette[1]);
    CHECK(strip.at<Vec3b>(5, 74) == palette[0]);
    CHECK(strip.at<Vec3b>(5, 76) == palette[1]);
    CHECK(toHex(Vec3b(0, 0, 255)) == "#FF0000");
    CHECK(toHex(Vec3b(1, 2, 3)) == "#030201");
}
