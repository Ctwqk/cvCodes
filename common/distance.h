// Distance / dissimilarity measures between feature vectors.
//
// Convention used everywhere in this repository: SMALLER means MORE SIMILAR,
// so results can be sorted ascending. (The original per-project copies mixed
// sign conventions, e.g. "negative SSD" and "1 - cos" side by side.)
// All functions throw std::invalid_argument when the vectors differ in length.
#pragma once
#include <vector>

namespace cvc {

// sum_i (a_i - b_i)^2
float sumSquaredDiff(const std::vector<float>& a, const std::vector<float>& b);

// sqrt(sumSquaredDiff)
float euclidean(const std::vector<float>& a, const std::vector<float>& b);

// sum_i |a_i - b_i|
float manhattan(const std::vector<float>& a, const std::vector<float>& b);

// 1 - cos(angle). 0 for identical directions, 1 for orthogonal, 2 for opposite.
// Two zero vectors have distance 0; a zero vs non-zero vector has distance 1.
float cosineDistance(const std::vector<float>& a, const std::vector<float>& b);

// 1 - sum_i min(a_i/|a|, b_i/|b|), i.e. histogram intersection after each
// histogram is normalised to unit mass (so images of different sizes are
// comparable). Inputs must be non-negative. Result is in [0, 1].
float histogramIntersectionDistance(const std::vector<float>& a, const std::vector<float>& b);

// Euclidean distance after dividing each coordinate by `scale[i]` (use the
// per-feature standard deviation for "scaled Euclidean"). Zero scales are
// treated as 1.
float scaledEuclidean(const std::vector<float>& a, const std::vector<float>& b,
                      const std::vector<float>& scale);

}  // namespace cvc
