#include "doctest.h"
#include "patternMatch.hpp"
#include "saveOrReadPattern.hpp"
#include "app_io.h"
#include <filesystem>
#include <fstream>

TEST_CASE("homography maps pattern into frame even with permuted descriptor indices") {
    std::vector<cv::KeyPoint> pattern, query;
    const std::vector<cv::Point2f> points{{0,0},{100,0},{100,100},{0,100},{40,60}};
    for (const auto& p : points) pattern.emplace_back(p, 1);
    std::vector<cv::DMatch> matches;
    for (int i : {3,0,4,1,2}) {
        matches.emplace_back(static_cast<int>(query.size()), i, 0.f);
        query.emplace_back(points[i] * 2 + cv::Point2f(17,31), 1);
    }
    cv::Mat H;
    REQUIRE(getHomoMat(pattern, query, matches, H) == 0);
    std::vector<cv::Point2f> result;
    cv::perspectiveTransform(points, result, H);
    for (size_t i=0; i<points.size(); ++i) CHECK(cv::norm(result[i] - (points[i]*2+cv::Point2f(17,31))) < 1e-3);
    matches.resize(3);
    CHECK(getHomoMat(pattern, query, matches, H) == -1);
    CHECK(H.empty());
}

TEST_CASE("SIFT database roundtrip, matching, and malformed input") {
    const auto path = (std::filesystem::temp_directory_path()/"pu-cvcodes-pattern-test.txt").string();
    REQUIRE(establishPatterns(cvc::repoPath("project1"), path, 100, 1) == 0);
    std::vector<std::vector<cv::KeyPoint>> points;
    std::vector<cv::Mat> descriptors, corners;
    std::vector<std::vector<cv::Point3f>> world;
    REQUIRE(readPatternsFromCsv(path, points, descriptors, corners, world) == 0);
    REQUIRE(descriptors.size() == 1);
    CHECK(corners[0].at<double>(0,2) > 1);
    CHECK(world[0][3].x <= 1);
    CHECK(world[0][3].y <= 1);
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    cv::Mat H;
    int target = -1;
    REQUIRE(matchPattern(descriptors, descriptors[0], points, points[0], matcher, matches, target, H, -1) == 0);
    CHECK(target == 0);
    CHECK(matches.size() >= 4);
    CHECK(cv::norm(H, cv::Mat::eye(3,3,CV_64F), cv::NORM_INF) < 1e-4);
    cv::Mat empty;
    CHECK(matchPattern(descriptors, empty, points, points[0], matcher, matches, target, H, -1) == -1);
    CHECK(target == -1);
    std::ofstream(path) << "-1 128 -1\n";
    CHECK(readPatternsFromCsv(path, points, descriptors, corners, world) == -1);
    std::filesystem::remove(path);
}
