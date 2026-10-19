/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Header file for implementation of a K-means algorithm
  (course skeleton; bug fixes and tests added in the portfolio upgrade)
*/
#ifndef KMEANS_H
#define KMEANS_H

#include <opencv2/core.hpp>
#include <vector>

// Sum of squared differences between two 3-channel colors (any element type).
template <typename A, typename B>
inline int ssd3(const A& a, const B& b) {
    const int d0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
    const int d1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
    const int d2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
    return d0 * d0 + d1 * d1 + d2 * d2;
}

// Kept for source compatibility with the course skeleton.
#define SSD(a, b) ssd3((a), (b))

/*
  data:          pixels to cluster
  means:         output, the K cluster means
  labels:        caller-allocated array with data.size() entries; receives the
                 index of the closest mean for every pixel
  K:             number of clusters (1 <= K <= data.size())
  maxIterations: maximum number of E-M iterations
  stopThresh:    stop when the summed squared movement of the means is <= this
  seed:          seed for the initial comb-sampling offset (deterministic)

  Returns 0 on success, -1 on invalid arguments. On return `labels` is
  consistent with `means` (it is recomputed after the last update).
 */
int kmeans(const std::vector<cv::Vec3b>& data, std::vector<cv::Vec3b>& means, int* labels, int K,
           int maxIterations = 10, int stopThresh = 0, unsigned seed = 0);

#endif
