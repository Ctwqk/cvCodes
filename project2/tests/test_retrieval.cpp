#include <cstdio>
#include <filesystem>
#include <set>

#include <opencv2/imgcodecs.hpp>

#include "doctest.h"
#include "imgfeatures.h"
#include "retrieval.h"

namespace fs = std::filesystem;
using cv::Mat;
using cv::Vec3b;

namespace {
cvc::FeatureTable makeTable() {
    cvc::FeatureTable t;
    t.names = {"a.jpg", "b.jpg", "c.jpg", "d.jpg"};
    t.data = {{0, 0}, {1, 0}, {3, 4}, {1, 0}};
    return t;
}
Mat solid(Vec3b c, int rows = 64, int cols = 64) { return Mat(rows, cols, CV_8UC3, cv::Scalar(c[0], c[1], c[2])); }
// A colour with Gaussian noise: gives histograms some spread so that nearby colours overlap.
Mat noisy(Vec3b c, int rows, int cols, unsigned seed) {
    Mat m = solid(c, rows, cols), n(rows, cols, CV_8UC3);
    cv::RNG rng(seed);
    rng.fill(n, cv::RNG::NORMAL, 0, 12);
    cv::add(m, n, m);
    return m;
}
}  // namespace

TEST_CASE("rankMatches orders by distance, breaks ties by row order and honours k") {
    const auto db = makeTable();
    const auto m = rankMatches({1, 0}, db, Metric::SSD, 3);
    REQUIRE(m.size() == 3);
    CHECK(m[0].name == "b.jpg");  // distance 0, first of the tie
    CHECK(m[1].name == "d.jpg");  // distance 0, second of the tie
    CHECK(m[2].name == "a.jpg");  // distance 1
    CHECK(m[0].distance == 0.f);
    CHECK(m[2].distance == 1.f);
    CHECK(rankMatches({1, 0}, db, Metric::SSD, 100).size() == 4);
    CHECK(rankMatches({1, 0}, db, Metric::SSD, 0).empty());
}

TEST_CASE("rankMatches skips the query itself (by base name) instead of blindly dropping rank 1") {
    const auto db = makeTable();
    const auto m = rankMatches({1, 0}, db, Metric::SSD, 4, "some/dir/b.jpg");
    for (const Match& x : m) CHECK(x.name != "b.jpg");
    CHECK(m.size() == 3);
    CHECK(m[0].name == "d.jpg");
    // a query that is not in the database keeps its true best match at rank 1
    CHECK(rankMatches({1, 0}, db, Metric::SSD, 1, "zzz.jpg")[0].name == "b.jpg");
}

TEST_CASE("rankMatches can list the farthest rows and ignores rows of the wrong length") {
    auto db = makeTable();
    db.names.push_back("short.jpg");
    db.data.push_back({1});
    const auto far = rankMatches({0, 0}, db, Metric::SSD, 1, "", true);
    REQUIRE(far.size() == 1);
    CHECK(far[0].name == "c.jpg");
    for (const Match& x : rankMatches({0, 0}, db, Metric::SSD, 10)) CHECK(x.name != "short.jpg");
}

TEST_CASE("every metric ranks an identical vector first") {
    cvc::FeatureTable t;
    t.names = {"x", "y", "z"};
    t.data = {{5, 1, 0, 2}, {0, 4, 4, 1}, {2, 2, 2, 2}};
    for (Metric metric : {Metric::SSD, Metric::HistIntersection, Metric::Cosine}) {
        const auto m = rankMatches(t.data[1], t, metric, 1);
        REQUIRE(m.size() == 1);
        CHECK(m[0].name == "y");
        CHECK(m[0].distance == doctest::Approx(0.f).epsilon(1e-6));
    }
}

TEST_CASE("feature registry") {
    std::set<std::string> names, files;
    for (const FeatureSpec& s : featureSpecs()) {
        CHECK(names.insert(s.name).second);
        CHECK(files.insert(s.dbFile).second);
        CHECK(findFeatureSpec(s.name) == &s);
    }
    CHECK(findFeatureSpec("nope") == nullptr);
    CHECK(findFeatureSpec("dnn")->dbFile == kDnnCsvName);
}

TEST_CASE("computeFeature produces the documented dimensions for every feature") {
    cv::RNG rng(5);
    Mat img(80, 96, CV_8UC3);
    rng.fill(img, cv::RNG::UNIFORM, 0, 256);
    cvc::FeatureTable dnn;
    dnn.names = {"pic.0001.jpg"};
    dnn.data = {std::vector<float>(512, 0.5f)};
    const std::vector<std::pair<std::string, size_t>> expected = {
        {"7x7central", 147},
        {"histogram", static_cast<size_t>(kHistSize)},
        {"multihist", static_cast<size_t>(kMultiHistRois * kHistSize)},
        {"texturecolor", static_cast<size_t>(kTextureSize + kHistSize)},
        {"gabor", static_cast<size_t>(kGaborSize)},
        {"dnn", 512},
        {"mycustom", static_cast<size_t>(kFourierSize + 512)},
    };
    for (const auto& [name, size] : expected) {
        std::vector<float> f;
        std::string err;
        INFO("feature " << name);
        REQUIRE_MESSAGE(computeFeature(*findFeatureSpec(name), img, "dir/pic.0001.jpg", &dnn, f, err), err);
        CHECK(f.size() == size);
    }
}

TEST_CASE("computeFeature explains missing embeddings and small images") {
    Mat img = solid(Vec3b(1, 2, 3));
    std::vector<float> f;
    std::string err;
    CHECK_FALSE(computeFeature(*findFeatureSpec("dnn"), img, "pic.0001.jpg", nullptr, f, err));
    CHECK(err.find(kDnnCsvName) != std::string::npos);
    cvc::FeatureTable dnn;
    dnn.names = {"pic.0002.jpg"};
    dnn.data = {{1, 2}};
    CHECK_FALSE(computeFeature(*findFeatureSpec("mycustom"), img, "pic.0001.jpg", &dnn, f, err));
    CHECK(err.find("pic.0001.jpg") != std::string::npos);
    CHECK_FALSE(computeFeature(*findFeatureSpec("7x7central"), solid(Vec3b(1, 2, 3), 5, 5), "x.jpg", nullptr, f, err));
    CHECK_FALSE(err.empty());
}

TEST_CASE("end to end: database of coloured images, query finds the similar colour first") {
    const fs::path dir = fs::temp_directory_path() / "cvc_test_p2_db";
    fs::remove_all(dir);
    fs::create_directories(dir / "imgs");
    const std::vector<std::pair<std::string, Vec3b>> colours = {
        {"red.png", Vec3b(25, 25, 225)},     {"red2.png", Vec3b(30, 28, 215)},
        {"green.png", Vec3b(30, 220, 40)},   {"blue.png", Vec3b(230, 40, 30)},
        {"yellow.png", Vec3b(20, 220, 220)},
    };
    unsigned seed = 1;
    for (const auto& [name, c] : colours) cv::imwrite((dir / "imgs" / name).string(), noisy(c, 48, 64, seed++));

    for (const char* feature : {"histogram", "multihist", "texturecolor", "gabor", "7x7central"}) {
        const FeatureSpec& spec = *findFeatureSpec(feature);
        std::string err;
        const std::string csv = (dir / spec.dbFile).string();
        REQUIRE_MESSAGE(buildDatabase(spec, (dir / "imgs").string(), csv, nullptr, 0, err) == 5, err);
        cvc::FeatureTable db;
        REQUIRE(cvc::readFeatureTable(csv, db, &err));
        REQUIRE(db.size() == 5);

        // query: a (not stored) red image of a different size
        std::vector<float> q;
        REQUIRE(computeFeature(spec, noisy(Vec3b(28, 26, 220), 96, 128, 99), "query.png", nullptr, q, err));
        const auto m = rankMatches(q, db, spec.metric, 2);
        INFO("feature " << std::string(feature));
        if (std::string(feature) != "gabor") {  // flat images carry no texture to tell colours apart
            REQUIRE(m.size() == 2);
            CHECK((m[0].name == "red.png" || m[0].name == "red2.png"));
            CHECK((m[1].name == "red.png" || m[1].name == "red2.png"));
        }
    }
    // --limit
    std::string err;
    CHECK(buildDatabase(*findFeatureSpec("histogram"), (dir / "imgs").string(), (dir / "limited.csv").string(), nullptr, 2, err) == 2);
    CHECK(buildDatabase(*findFeatureSpec("histogram"), (dir / "empty").string(), (dir / "x.csv").string(), nullptr, 0, err) == -1);
    CHECK_FALSE(err.empty());
    fs::remove_all(dir);
}

TEST_CASE("makeMontage has one tile for the query and one per match") {
    std::vector<Mat> imgs = {solid(Vec3b(0, 0, 255)), solid(Vec3b(0, 255, 0))};
    std::vector<Match> matches = {{0, "a.jpg", 0.1f}, {1, "b.jpg", 0.2f}};
    const Mat m = makeMontage(solid(Vec3b(255, 0, 0)), "q.jpg", imgs, matches, 100, 80);
    CHECK(m.cols == 300);
    CHECK(m.rows > 80);
    CHECK(m.at<Vec3b>(40, 50) == Vec3b(255, 0, 0));
    CHECK(m.at<Vec3b>(40, 150) == Vec3b(0, 0, 255));
    CHECK(m.at<Vec3b>(40, 250) == Vec3b(0, 255, 0));
}
