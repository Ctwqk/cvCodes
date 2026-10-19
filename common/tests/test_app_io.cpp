#include <filesystem>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "doctest.h"

using namespace cvc;
namespace fs = std::filesystem;

namespace {
Args parse(std::vector<const char*> argv, std::set<std::string> flags = {}) {
    return Args(static_cast<int>(argv.size()), argv.data(), std::move(flags));
}
}  // namespace

TEST_CASE("Args: options, flags, positionals and defaults") {
    const Args a = parse({"prog", "in.jpg", "--k", "8", "--seed=7", "--headless", "--ratio", "0.5",
                          "extra", "--neg", "-1"},
                         {"headless"});
    CHECK(a.program() == "prog");
    CHECK(a.positional() == std::vector<std::string>{"in.jpg", "extra"});
    CHECK(a.getInt("k", 0) == 8);
    CHECK(a.getInt("seed", 0) == 7);
    CHECK(a.getDouble("ratio", 0) == doctest::Approx(0.5));
    CHECK(a.getInt("neg", 0) == -1);
    CHECK(a.has("headless"));
    CHECK_FALSE(a.has("missing"));
    CHECK(a.getInt("missing", 42) == 42);
    CHECK(a.get("missing", "dflt") == "dflt");
}

TEST_CASE("Args: a flag followed by another option does not swallow it") {
    const Args a = parse({"prog", "--verbose", "--out", "dir"});
    CHECK(a.has("verbose"));
    CHECK(a.get("out") == "dir");
}

TEST_CASE("Args: malformed numbers throw") {
    const Args a = parse({"prog", "--k", "eight", "--x", "1.5.2"});
    CHECK_THROWS_AS(a.getInt("k", 0), std::invalid_argument);
    CHECK_THROWS_AS(a.getDouble("x", 0), std::invalid_argument);
}

TEST_CASE("listImages is sorted, filters by extension, tolerates missing dirs") {
    const fs::path dir = fs::temp_directory_path() / "cvc_test_listimages";
    fs::remove_all(dir);
    fs::create_directories(dir);
    cv::Mat img(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    for (const char* n : {"b.png", "a.jpg", "C.PNG"}) cv::imwrite((dir / n).string(), img);
    std::ofstream((dir / "notes.txt").string()) << "x";
    const auto files = listImages(dir.string());
    REQUIRE(files.size() == 3);
    CHECK(fs::path(files[0]).filename() == "C.PNG");  // uppercase sorts first
    CHECK(fs::path(files[1]).filename() == "a.jpg");
    CHECK(fs::path(files[2]).filename() == "b.png");
    CHECK(listImages((dir / "nope").string()).empty());
    fs::remove_all(dir);
}

TEST_CASE("Viewer in headless mode writes numbered PNGs") {
    const fs::path dir = fs::temp_directory_path() / "cvc_test_viewer";
    fs::remove_all(dir);
    Viewer v(true, dir.string(), 2);
    cv::Mat img(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    for (int i = 0; i < 5; ++i) v.show("win", img);
    CHECK(v.savedCount() == 3);  // frames 0, 2, 4
    CHECK(fs::exists(dir / "win_00000.png"));
    CHECK_FALSE(fs::exists(dir / "win_00001.png"));
    CHECK(fs::exists(dir / "win_00004.png"));
    CHECK(v.waitKey(1000) == -1);
    fs::remove_all(dir);
}

TEST_CASE("openSource opens an image file and reports bad sources") {
    const fs::path dir = fs::temp_directory_path() / "cvc_test_source";
    fs::create_directories(dir);
    const std::string p = (dir / "x.png").string();
    cv::imwrite(p, cv::Mat(6, 5, CV_8UC3, cv::Scalar(0, 255, 0)));
    cv::VideoCapture cap = openSource(p);
    cv::Mat frame;
    REQUIRE(cap.read(frame));
    CHECK(frame.cols == 5);
    CHECK(frame.rows == 6);
    CHECK_THROWS_AS(openSource((dir / "missing.avi").string()), std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("repoPath joins onto the repository root") {
    CHECK(fs::exists(repoPath("CMakeLists.txt")));
}
