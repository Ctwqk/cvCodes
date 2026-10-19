#include <cstdio>
#include <filesystem>
#include <fstream>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "doctest.h"
#include "filterviz.h"
#include "idx.h"
#include "npy.h"

using namespace p5;
namespace fs = std::filesystem;

namespace {
std::string tmp(const char* name) { return (fs::temp_directory_path() / (std::string("cvc_test_p5_") + name)).string(); }

void writeBE32(std::ostream& o, uint32_t v) {
    const unsigned char b[4] = {static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
                                static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v)};
    o.write(reinterpret_cast<const char*>(b), 4);
}
}  // namespace

TEST_CASE("npy: write then read gives the same shape and data") {
    const std::string path = tmp("roundtrip.npy");
    std::vector<float> data(2 * 3 * 4);
    for (size_t i = 0; i < data.size(); i++) data[i] = static_cast<float>(i) * 0.5f - 3.f;
    std::string err;
    REQUIRE_MESSAGE(writeNpy(path, {2, 3, 4}, data, &err), err);
    NpyArray a;
    REQUIRE_MESSAGE(readNpy(path, a, &err), err);
    CHECK(a.shape == std::vector<size_t>{2, 3, 4});
    CHECK(a.data == data);
    // header is padded to a multiple of 64 bytes, as numpy requires
    CHECK((fs::file_size(path) - data.size() * sizeof(float)) % 64 == 0);
    std::remove(path.c_str());
}

TEST_CASE("npy: 1-D shapes use the trailing-comma tuple, size mismatches are rejected") {
    const std::string path = tmp("oned.npy");
    REQUIRE(writeNpy(path, {5}, {1, 2, 3, 4, 5}));
    std::ifstream f(path, std::ios::binary);
    std::string header((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(header.find("'shape': (5,)") != std::string::npos);
    NpyArray a;
    REQUIRE(readNpy(path, a));
    CHECK(a.shape == std::vector<size_t>{5});
    std::string err;
    CHECK_FALSE(writeNpy(path, {2, 2}, {1, 2, 3}, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(writeNpy("/definitely/not/here/x.npy", {1}, {1}, &err));
    std::remove(path.c_str());
}

TEST_CASE("npy: reads the files numpy wrote for this project") {
    NpyArray filters, image;
    std::string err;
    REQUIRE_MESSAGE(readNpy(cvc::repoPath("project5/filters.npy"), filters, &err), err);
    REQUIRE_MESSAGE(readNpy(cvc::repoPath("project5/image.npy"), image, &err), err);
    CHECK(filters.shape == std::vector<size_t>{10, 5, 5});
    CHECK(image.shape == std::vector<size_t>{28, 28});
    float lo = 1e9f, hi = -1e9f;
    for (float v : image.data) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    CHECK(lo == 0.f);
    CHECK(hi == doctest::Approx(1.f));
}

TEST_CASE("npy: other dtypes and bad files") {
    // hand-made '<f8' and '|u1' arrays
    auto writeRaw = [](const std::string& path, const std::string& descr, const std::string& shape, const std::string& payload) {
        std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
        header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
        header += '\n';
        std::ofstream o(path, std::ios::binary);
        o.write("\x93NUMPY\x01\x00", 8);
        const unsigned char len[2] = {static_cast<unsigned char>(header.size() & 0xff), static_cast<unsigned char>(header.size() >> 8)};
        o.write(reinterpret_cast<const char*>(len), 2);
        o << header << payload;
    };
    const std::string path = tmp("dtypes.npy");
    const double d[2] = {1.5, -2.0};
    writeRaw(path, "<f8", "(2,)", std::string(reinterpret_cast<const char*>(d), sizeof d));
    NpyArray a;
    REQUIRE(readNpy(path, a));
    CHECK(a.data == std::vector<float>{1.5f, -2.f});
    writeRaw(path, "|u1", "(2, 2)", std::string("\x01\x02\x03\xff", 4));
    REQUIRE(readNpy(path, a));
    CHECK(a.data == std::vector<float>{1, 2, 3, 255});
    writeRaw(path, "<f4", "()", std::string("\x00\x00\x80\x3f", 4));  // 0-d array
    REQUIRE(readNpy(path, a));
    CHECK(a.shape.empty());
    CHECK(a.data == std::vector<float>{1.f});

    std::string err;
    writeRaw(path, "<f4", "(4,)", std::string(8, '\0'));  // fewer bytes than the shape needs
    CHECK_FALSE(readNpy(path, a, &err));
    CHECK(err.find("truncated") != std::string::npos);
    writeRaw(path, "<f4", "(18446744073709551615, 2)", "");
    CHECK_FALSE(readNpy(path, a, &err));
    writeRaw(path, "<f4", "(-2,)", std::string(8, '\0'));
    CHECK_FALSE(readNpy(path, a, &err));
    writeRaw(path, "<c16", "(1,)", std::string(16, '\0'));
    CHECK_FALSE(readNpy(path, a, &err));
    std::ofstream(path) << "this is not numpy";
    CHECK_FALSE(readNpy(path, a, &err));
    CHECK_FALSE(readNpy(tmp("missing.npy"), a, &err));
    std::remove(path.c_str());
}

TEST_CASE("idx: images and labels") {
    const std::string imgPath = tmp("imgs.idx"), lblPath = tmp("lbls.idx");
    {
        std::ofstream o(imgPath, std::ios::binary);
        writeBE32(o, 0x803);
        writeBE32(o, 3);
        writeBE32(o, 2);
        writeBE32(o, 4);
        for (int i = 0; i < 3 * 2 * 4; i++) o.put(static_cast<char>(i * 10));
        std::ofstream l(lblPath, std::ios::binary);
        writeBE32(l, 0x801);
        writeBE32(l, 3);
        l.put(7);
        l.put(0);
        l.put(9);
    }
    std::vector<cv::Mat> imgs;
    std::vector<uint8_t> lbls;
    std::string err;
    REQUIRE_MESSAGE(readIdxImages(imgPath, imgs, &err), err);
    REQUIRE(imgs.size() == 3);
    CHECK(imgs[0].size() == cv::Size(4, 2));
    CHECK(imgs[1].at<uchar>(0, 0) == 80);
    CHECK(imgs[2].at<uchar>(1, 3) == 230);
    REQUIRE(readIdxImages(imgPath, imgs, &err, 2));
    CHECK(imgs.size() == 2);
    REQUIRE_MESSAGE(readIdxLabels(lblPath, lbls, &err), err);
    CHECK(lbls == std::vector<uint8_t>{7, 0, 9});
    REQUIRE(readIdxLabels(lblPath, lbls, &err, 1));
    CHECK(lbls.size() == 1);
    // wrong magic, truncated payload, missing file
    CHECK_FALSE(readIdxImages(lblPath, imgs, &err));
    CHECK_FALSE(readIdxLabels(imgPath, lbls, &err));
    {
        std::ofstream o(imgPath, std::ios::binary | std::ios::trunc);
        writeBE32(o, 0x803);
        writeBE32(o, 5);
        writeBE32(o, 2);
        writeBE32(o, 2);
        o.write("abcdefgh", 8);  // two images instead of five
    }
    CHECK_FALSE(readIdxImages(imgPath, imgs, &err));
    CHECK(err.find("ends after 2") != std::string::npos);
    CHECK_FALSE(readIdxImages(tmp("none.idx"), imgs, &err));
    std::remove(imgPath.c_str());
    std::remove(lblPath.c_str());
}

TEST_CASE("the shipped MNIST test files parse: 10000 images of 28x28 and 10000 labels") {
    std::vector<cv::Mat> imgs;
    std::vector<uint8_t> lbls;
    std::string err;
    const std::string dir = cvc::repoPath("project5/data/MNIST/raw");
    REQUIRE_MESSAGE(readIdxImages(dir + "/t10k-images-idx3-ubyte", imgs, &err), err);
    REQUIRE_MESSAGE(readIdxLabels(dir + "/t10k-labels-idx1-ubyte", lbls, &err), err);
    CHECK(imgs.size() == 10000);
    CHECK(lbls.size() == 10000);
    CHECK(imgs[0].size() == cv::Size(28, 28));
    CHECK(lbls[0] == 7);  // the first test digit is a 7
}

TEST_CASE("applyFilters: identity kernel, shifted kernel and shape validation") {
    NpyArray filters, image;
    image.shape = {6, 7};
    image.data.resize(42);
    for (size_t i = 0; i < 42; i++) image.data[i] = static_cast<float>(i % 7) + 0.1f * static_cast<float>(i / 7);
    filters.shape = {2, 3, 3};
    filters.data.assign(18, 0.f);
    filters.data[4] = 1.f;       // identity (centre of the first 3x3)
    filters.data[9 + 5] = 1.f;   // second kernel: picks the pixel to the right
    std::vector<cv::Mat> responses, kernels;
    cv::Mat img;
    REQUIRE(applyFilters(filters, image, responses, &kernels, &img));
    REQUIRE(responses.size() == 2);
    CHECK(responses[0].size() == cv::Size(7, 6));
    CHECK(cv::norm(responses[0], img, cv::NORM_INF) < 1e-6);
    CHECK(responses[1].at<float>(2, 3) == doctest::Approx(img.at<float>(2, 4)));  // correlation: right neighbour
    CHECK(kernels[1].at<float>(1, 2) == 1.f);

    NpyArray bad = filters;
    bad.shape = {2, 3};  // not 3-D
    CHECK_FALSE(applyFilters(bad, image, responses));
    bad = filters;
    bad.data.pop_back();
    CHECK_FALSE(applyFilters(bad, image, responses));
    NpyArray img3 = image;
    img3.shape = {6, 7, 1};
    CHECK_FALSE(applyFilters(filters, img3, responses));
}

TEST_CASE("applyFilters on the shipped network filters and digit; the contact sheet has the expected size") {
    NpyArray filters, image;
    REQUIRE(readNpy(cvc::repoPath("project5/filters.npy"), filters));
    REQUIRE(readNpy(cvc::repoPath("project5/image.npy"), image));
    std::vector<cv::Mat> responses, kernels;
    cv::Mat img;
    REQUIRE(applyFilters(filters, image, responses, &kernels, &img));
    CHECK(responses.size() == 10);
    CHECK(responses[0].size() == cv::Size(28, 28));
    const cv::Mat sheet = filterSheet(img, kernels, responses, 5, 100);
    CHECK(sheet.cols == 500);
    CHECK(sheet.type() == CV_8UC3);
    CHECK(sheet.rows > 400);
}
