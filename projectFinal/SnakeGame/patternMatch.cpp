/* Taiwei Cui, April 2024: match descriptors and estimate a planar homography.
 * DMatch queryIdx indexes the frame; trainIdx indexes the stored pattern.
 * Homographies map pattern coordinates to frame coordinates.
 */
#include "patternMatch.hpp"
using namespace cv;
using namespace std;

int getHomoMat(vector<KeyPoint>& pattern, vector<KeyPoint>& query, vector<DMatch>& matches, Mat& H) {
    H.release();
    if (matches.size() < 4) return -1;
    vector<Point2f> from, to;
    for (const auto& m : matches) {
        if (m.trainIdx < 0 || m.queryIdx < 0 || static_cast<size_t>(m.trainIdx) >= pattern.size() ||
            static_cast<size_t>(m.queryIdx) >= query.size()) return -1;
        from.push_back(pattern[m.trainIdx].pt);
        to.push_back(query[m.queryIdx].pt);
    }
    Mat mask;
    H = findHomography(from, to, RANSAC, 3, mask);
    if (H.empty()) return -1;
    vector<DMatch> inliers;
    for (size_t i = 0; i < matches.size(); ++i) if (mask.at<uchar>(static_cast<int>(i))) inliers.push_back(matches[i]);
    matches = std::move(inliers);
    return matches.size() >= 4 ? 0 : -1;
}

int getHomoMat(vector<Point2f>& pattern, vector<Point2f>& query, Mat& H) {
    H.release();
    if (pattern.size() < 4 || pattern.size() != query.size()) return -1;
    H = findHomography(pattern, query, RANSAC, 3);
    return H.empty() ? -1 : 0;
}

int matchPattern(vector<Mat>& descriptors, Mat& queryDescriptor, vector<vector<KeyPoint>>& patternPoints,
                 vector<KeyPoint>& queryPoints, BFMatcher& matcher, vector<DMatch>& matches,
                 int& target, Mat& H, int idx) {
    target = -1;
    matches.clear();
    H.release();
    if (queryDescriptor.empty() || descriptors.size() != patternPoints.size()) return -1;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        if (idx >= 0 && i != static_cast<size_t>(idx)) continue;
        if (descriptors[i].rows < 2) continue;
        vector<vector<DMatch>> nearest;
        matcher.knnMatch(queryDescriptor, descriptors[i], nearest, 2);
        vector<DMatch> good;
        for (const auto& pair : nearest)
            if (pair.size() == 2 && pair[0].distance < 0.75f * pair[1].distance) good.push_back(pair[0]);
        Mat candidate;
        if (getHomoMat(patternPoints[i], queryPoints, good, candidate) == 0 && good.size() > matches.size()) {
            target = static_cast<int>(i);
            matches = std::move(good);
            H = candidate;
        }
    }
    return target >= 0 ? 0 : -1;
}
