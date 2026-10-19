#include "region.h"

#include <algorithm>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace {

// Union-find over pixel indices (the original allocated a 2-int array per pixel with `new` and
// never freed it: ~5 MB leaked per 640x480 frame).
struct DisjointSet {
    std::vector<int> parent;
    std::vector<unsigned char> rank;
    explicit DisjointSet(size_t n) : parent(n), rank(n, 0) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
    }
};

}  // namespace

int labelRegions(const cv::Mat& binary, cv::Mat& labels, std::vector<int>& areas) {
    areas.clear();
    labels = cv::Mat::zeros(binary.size(), CV_32S);
    if (binary.empty() || binary.type() != CV_8UC1) return 0;
    const int row = binary.rows, col = binary.cols;
    DisjointSet sets(binary.total());
    for (int i = 0; i < row; i++) {
        const uchar* p = binary.ptr<uchar>(i);
        for (int j = 0; j < col; j++) {
            if (!p[j]) continue;
            const int idx = i * col + j;
            if (j + 1 < col && p[j + 1]) sets.unite(idx, idx + 1);
            if (i + 1 < row && binary.at<uchar>(i + 1, j)) sets.unite(idx, idx + col);
        }
    }
    // number the roots in raster order, so ids are deterministic
    std::vector<int> id(binary.total(), 0);
    int count = 0;
    for (int i = 0; i < row; i++) {
        const uchar* p = binary.ptr<uchar>(i);
        int* l = labels.ptr<int>(i);
        for (int j = 0; j < col; j++) {
            if (!p[j]) continue;
            const int root = sets.find(i * col + j);
            if (!id[root]) {
                id[root] = ++count;
                areas.push_back(0);
            }
            l[j] = id[root];
            areas[id[root] - 1]++;
        }
    }
    return count;
}

std::vector<cv::Mat> regionize(const cv::Mat& src, cv::Mat& dst, int num, int minArea) {
    std::vector<cv::Mat> masks;
    if (src.empty() || src.type() != CV_8UC1) {
        dst = cv::Mat();
        return masks;
    }
    num = std::max(0, std::min(num, kMaxRegions));
    cv::Mat labels;
    std::vector<int> areas;
    const int count = labelRegions(src, labels, areas);

    // A region that touches the border band is background clutter (table, paper, ...): drop it.
    // The original flood-filled inwards from the band with a fixed-size ring buffer; labelling first
    // and discarding whole components gives the same set of regions without the overflow case.
    std::vector<unsigned char> touchesBorder(static_cast<size_t>(count) + 1, 0);
    for (int i = 0; i < src.rows; i++) {
        const int* l = labels.ptr<int>(i);
        const bool inBand = i < kEdgeBand || i >= src.rows - kEdgeBand;
        for (int j = 0; j < src.cols; j++) {
            if (l[j] && (inBand || j < kEdgeBand || j >= src.cols - kEdgeBand)) touchesBorder[l[j]] = 1;
        }
    }

    // candidates sorted by area (largest first); labels are numbered in raster order so ties are
    // broken by the first pixel of the region
    std::vector<int> order;
    for (int c = 1; c <= count; c++) {
        if (!touchesBorder[c] && areas[c - 1] >= minArea) order.push_back(c);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return areas[a - 1] > areas[b - 1]; });
    if (static_cast<int>(order.size()) > num) order.resize(static_cast<size_t>(num));

    cv::Mat planes[kMaxRegions];
    for (cv::Mat& p : planes) p = cv::Mat::zeros(src.size(), CV_8U);
    for (size_t k = 0; k < order.size(); k++) {
        cv::compare(labels, order[k], planes[k], cv::CMP_EQ);
        masks.push_back(planes[k].clone());
    }
    // colour the regions in order of Blue, Green, Red, Cyan, Yellow, Magenta
    for (int i = 0; i < 3; i++) {
        cv::add(planes[3 + i], planes[i], planes[i]);
        cv::add(planes[3 + i], planes[(i + 1) % 3], planes[(i + 1) % 3]);
    }
    cv::merge(planes, 3, dst);
    return masks;
}
