#include <cmath>
#include <stdexcept>

#include "distance.h"
#include "doctest.h"

using namespace cvc;
using V = std::vector<float>;

TEST_CASE("sumSquaredDiff / euclidean / manhattan") {
    const V a = {1, 2, 3}, b = {4, 6, 3};
    CHECK(sumSquaredDiff(a, b) == doctest::Approx(25.f));
    CHECK(euclidean(a, b) == doctest::Approx(5.f));
    CHECK(manhattan(a, b) == doctest::Approx(7.f));
    CHECK(euclidean(a, a) == 0.f);
    CHECK(euclidean(a, b) == euclidean(b, a));
}

TEST_CASE("cosineDistance") {
    CHECK(cosineDistance({1, 0}, {2, 0}) == doctest::Approx(0.f).epsilon(1e-6));
    CHECK(cosineDistance({1, 0}, {0, 3}) == doctest::Approx(1.f));
    CHECK(cosineDistance({1, 0}, {-1, 0}) == doctest::Approx(2.f));
    CHECK(cosineDistance({0, 0}, {0, 0}) == 0.f);
    CHECK(cosineDistance({0, 0}, {1, 2}) == 1.f);
    // scale invariant
    CHECK(cosineDistance({1, 2, 3}, {2, 4, 6}) == doctest::Approx(0.f).epsilon(1e-6));
    // never negative due to rounding
    CHECK(cosineDistance({0.1f, 0.2f, 0.3f}, {0.1f, 0.2f, 0.3f}) >= 0.f);
}

TEST_CASE("histogramIntersectionDistance is size invariant and bounded") {
    const V h1 = {10, 20, 30, 40};
    const V h2 = {20, 40, 60, 80};  // same shape, twice the mass
    CHECK(histogramIntersectionDistance(h1, h2) == doctest::Approx(0.f).epsilon(1e-6));
    CHECK(histogramIntersectionDistance({1, 0}, {0, 1}) == doctest::Approx(1.f));
    CHECK(histogramIntersectionDistance({1, 1}, {1, 0}) == doctest::Approx(0.5f));
    CHECK(histogramIntersectionDistance({0, 0}, {0, 0}) == 0.f);
    CHECK(histogramIntersectionDistance({0, 0}, {1, 0}) == 1.f);
    CHECK_THROWS_AS(histogramIntersectionDistance({-1, 2}, {1, 1}), std::invalid_argument);
}

TEST_CASE("scaledEuclidean divides by the per-feature scale") {
    CHECK(scaledEuclidean({0, 0}, {2, 10}, {2, 10}) == doctest::Approx(std::sqrt(2.f)));
    CHECK(scaledEuclidean({0, 0}, {3, 4}, {0, 0}) == doctest::Approx(5.f));  // zero scale -> 1
}

TEST_CASE("size mismatch throws instead of returning a misleading score") {
    CHECK_THROWS_AS(euclidean({1, 2}, {1}), std::invalid_argument);
    CHECK_THROWS_AS(cosineDistance({1}, {1, 2}), std::invalid_argument);
    CHECK_THROWS_AS(histogramIntersectionDistance({1}, {1, 2}), std::invalid_argument);
    CHECK_THROWS_AS(scaledEuclidean({1, 2}, {1, 2}, {1}), std::invalid_argument);
}

TEST_CASE("smaller means more similar: ranking a small database") {
    const V query = {1, 0, 0};
    const std::vector<V> db = {{0, 1, 0}, {0.9f, 0.1f, 0}, {1, 0, 0}};
    int best = -1;
    float bestD = 1e9f;
    for (size_t i = 0; i < db.size(); ++i) {
        const float d = cosineDistance(query, db[i]);
        if (d < bestD) {
            bestD = d;
            best = static_cast<int>(i);
        }
    }
    CHECK(best == 2);
}
