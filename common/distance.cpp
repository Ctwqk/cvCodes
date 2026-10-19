#include "distance.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace cvc {

namespace {
void requireSameSize(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("feature size mismatch: " + std::to_string(a.size()) + " vs " +
                                    std::to_string(b.size()));
    }
}
}  // namespace

float sumSquaredDiff(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameSize(a, b);
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        s += d * d;
    }
    return static_cast<float>(s);
}

float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    return std::sqrt(sumSquaredDiff(a, b));
}

float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameSize(a, b);
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += std::fabs(static_cast<double>(a[i]) - b[i]);
    return static_cast<float>(s);
}

float cosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameSize(a, b);
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0 && nb == 0) return 0.f;
    if (na == 0 || nb == 0) return 1.f;
    double c = dot / (std::sqrt(na) * std::sqrt(nb));
    if (c > 1) c = 1;
    if (c < -1) c = -1;
    return static_cast<float>(1.0 - c);
}

float histogramIntersectionDistance(const std::vector<float>& a, const std::vector<float>& b) {
    requireSameSize(a, b);
    double sa = 0, sb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] < 0 || b[i] < 0) throw std::invalid_argument("histogram bins must be non-negative");
        sa += a[i];
        sb += b[i];
    }
    if (sa == 0 && sb == 0) return 0.f;
    if (sa == 0 || sb == 0) return 1.f;
    double inter = 0;
    for (size_t i = 0; i < a.size(); ++i) inter += std::fmin(a[i] / sa, b[i] / sb);
    double d = 1.0 - inter;
    if (d < 0) d = 0;  // rounding
    return static_cast<float>(d);
}

float scaledEuclidean(const std::vector<float>& a, const std::vector<float>& b,
                      const std::vector<float>& scale) {
    requireSameSize(a, b);
    requireSameSize(a, scale);
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double sc = scale[i] == 0 ? 1.0 : scale[i];
        const double d = (static_cast<double>(a[i]) - b[i]) / sc;
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}

}  // namespace cvc
