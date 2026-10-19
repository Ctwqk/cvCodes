#include <cstdio>
#include <filesystem>
#include <fstream>

#include "csv_util.h"
#include "doctest.h"

using namespace cvc;
namespace fs = std::filesystem;

namespace {
std::string tmpPath(const char* name) {
    return (fs::temp_directory_path() / (std::string("cvc_test_") + name)).string();
}
}  // namespace

TEST_CASE("formatFeatureLine / parseFeatureLine round trip") {
    const std::vector<float> v = {0.f, 1.5f, -2.25f, 1e-5f, 123456.f};
    const std::string line = formatFeatureLine("pic.0001.jpg", v);
    std::string name;
    std::vector<float> back;
    REQUIRE(parseFeatureLine(line, name, back));
    CHECK(name == "pic.0001.jpg");
    REQUIRE(back.size() == v.size());
    for (size_t i = 0; i < v.size(); ++i) CHECK(back[i] == doctest::Approx(v[i]).epsilon(1e-6));
}

TEST_CASE("parseFeatureLine rejects malformed input") {
    std::string name, err;
    std::vector<float> v;
    CHECK_FALSE(parseFeatureLine(",1,2", name, v, &err));
    CHECK_FALSE(parseFeatureLine("a,1,,3", name, v, &err));
    CHECK_FALSE(parseFeatureLine("a,1,x", name, v, &err));
    CHECK_FALSE(parseFeatureLine("a,1,2,", name, v, &err));
    CHECK(parseFeatureLine("a", name, v));  // name only: empty feature vector
    CHECK(v.empty());
    CHECK(parseFeatureLine("a,1,2\r", name, v));  // CRLF tolerated
    CHECK(v.size() == 2);
}

TEST_CASE("append + read back, with truncate") {
    const std::string path = tmpPath("append.csv");
    std::remove(path.c_str());
    REQUIRE(appendFeatureRow(path, "a", {1, 2, 3}, true));
    REQUIRE(appendFeatureRow(path, "b", {4, 5, 6}));
    FeatureTable t;
    std::string err;
    REQUIRE_MESSAGE(readFeatureTable(path, t, &err), err);
    REQUIRE(t.size() == 2);
    CHECK(t.names[1] == "b");
    CHECK(t.data[1] == std::vector<float>{4, 5, 6});

    REQUIRE(appendFeatureRow(path, "only", {9}, true));  // truncate resets the file
    REQUIRE(readFeatureTable(path, t));
    CHECK(t.size() == 1);
    std::remove(path.c_str());
}

TEST_CASE("rows with commas in the name are rejected") {
    const std::string path = tmpPath("badname.csv");
    std::string err;
    CHECK_FALSE(appendFeatureRow(path, "a,b", {1}, true, &err));
    CHECK_FALSE(err.empty());
    std::remove(path.c_str());
}

TEST_CASE("long names and very long rows (the old fixed 256-byte buffers) work") {
    const std::string path = tmpPath("long.csv");
    const std::string longName(1000, 'x');
    std::vector<float> v(5000);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i) * 0.5f;
    REQUIRE(appendFeatureRow(path, longName, v, true));
    FeatureTable t;
    REQUIRE(readFeatureTable(path, t));
    REQUIRE(t.size() == 1);
    CHECK(t.names[0] == longName);
    CHECK(t.data[0] == v);
    std::remove(path.c_str());
}

TEST_CASE("readFeatureTable: blank lines skipped, missing final newline, errors report the line") {
    const std::string path = tmpPath("misc.csv");
    {
        std::ofstream o(path);
        o << "a,1,2\n\nb,3,4";  // no trailing newline, blank line in the middle
    }
    FeatureTable t;
    REQUIRE(readFeatureTable(path, t));
    CHECK(t.size() == 2);
    {
        std::ofstream o(path);
        o << "a,1,2\nb,oops\n";
    }
    std::string err;
    CHECK_FALSE(readFeatureTable(path, t, &err));
    CHECK(err.find(":2:") != std::string::npos);
    CHECK_FALSE(readFeatureTable(tmpPath("does_not_exist.csv"), t, &err));
    std::remove(path.c_str());
}

TEST_CASE("writeFeatureTable validates sizes") {
    FeatureTable t;
    t.names = {"a", "b"};
    t.data = {{1}};
    CHECK_FALSE(writeFeatureTable(tmpPath("w.csv"), t));
    t.data = {{1}, {2, 3}};
    REQUIRE(writeFeatureTable(tmpPath("w.csv"), t));
    FeatureTable r;
    REQUIRE(readFeatureTable(tmpPath("w.csv"), r));
    CHECK(r.data[1].size() == 2);
    std::remove(tmpPath("w.csv").c_str());
}

TEST_CASE("baseName") {
    CHECK(baseName("a/b/pic.jpg") == "pic.jpg");
    CHECK(baseName("pic.jpg") == "pic.jpg");
    CHECK(baseName("a\\b\\pic.jpg") == "pic.jpg");
}
