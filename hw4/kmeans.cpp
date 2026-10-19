/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Implementation of a K-means algorithm.

  Portfolio-upgrade fixes compared to the course skeleton:
   - the comb-sampling offset did `rand() % (size % K)`, a division by zero
     whenever the number of samples was a multiple of K;
   - an empty cluster divided by a zero count; it now keeps its previous mean;
   - means are rounded instead of truncated;
   - labels are recomputed after the last mean update so they always match
     the returned means;
   - the random offset comes from a seeded generator (reproducible runs).
 */
#include "kmeans.h"

#include <cstdio>
#include <random>

int kmeans(const std::vector<cv::Vec3b>& data, std::vector<cv::Vec3b>& means, int* labels, int K,
           int maxIterations, int stopThresh, unsigned seed) {
    if (K < 1 || data.empty() || static_cast<size_t>(K) > data.size() || labels == nullptr) {
        std::fprintf(stderr, "kmeans: need 1 <= K <= number of data points (K=%d, n=%zu)\n", K,
                     data.size());
        return -1;
    }

    // Initialise the K means with comb sampling: evenly spaced samples, with a
    // random offset that keeps every index inside the data.
    means.clear();
    const size_t n = data.size();
    const size_t delta = n / static_cast<size_t>(K);
    const size_t rem = n % static_cast<size_t>(K);
    std::mt19937 rng(seed);
    const size_t istep = rem > 0 ? static_cast<size_t>(rng() % rem) : 0;
    for (int i = 0; i < K; ++i) {
        means.push_back(data[(istep + static_cast<size_t>(i) * delta) % n]);
    }

    auto classify = [&]() {
        for (size_t j = 0; j < n; ++j) {
            int minssd = ssd3(means[0], data[j]);
            int minidx = 0;
            for (int k = 1; k < K; ++k) {
                const int tssd = ssd3(means[k], data[j]);
                if (tssd < minssd) {
                    minssd = tssd;
                    minidx = k;
                }
            }
            labels[j] = minidx;
        }
    };

    for (int it = 0; it < maxIterations; ++it) {
        classify();

        // Update step: accumulate sums and counts per cluster.
        std::vector<cv::Vec4i> tmeans(static_cast<size_t>(K), cv::Vec4i(0, 0, 0, 0));
        for (size_t j = 0; j < n; ++j) {
            cv::Vec4i& t = tmeans[static_cast<size_t>(labels[j])];
            t[0] += data[j][0];
            t[1] += data[j][1];
            t[2] += data[j][2];
            t[3]++;
        }

        int moved = 0;
        for (int k = 0; k < K; ++k) {
            cv::Vec4i& t = tmeans[static_cast<size_t>(k)];
            if (t[3] == 0) continue;  // empty cluster: keep the old mean
            const cv::Vec3b updated(static_cast<uchar>((t[0] + t[3] / 2) / t[3]),
                                    static_cast<uchar>((t[1] + t[3] / 2) / t[3]),
                                    static_cast<uchar>((t[2] + t[3] / 2) / t[3]));
            moved += ssd3(updated, means[static_cast<size_t>(k)]);
            means[static_cast<size_t>(k)] = updated;
        }
        if (moved <= stopThresh) break;
    }

    classify();  // labels consistent with the final means
    return 0;
}
