#include "pipeline.h"

#include <opencv2/imgproc.hpp>

#include "denoise.h"
#include "regionfeatures.h"
#include "threshold.h"

namespace {
// Fraction of foreground pixels in the outermost 2-pixel frame of a binary image.
double borderForeground(const cv::Mat& bin) {
    const int b = std::min(2, std::min(bin.rows, bin.cols) / 2);
    if (b <= 0) return 0;
    long fg = 0, total = 0;
    for (int i = 0; i < bin.rows; i++) {
        const uchar* p = bin.ptr<uchar>(i);
        const bool edgeRow = i < b || i >= bin.rows - b;
        for (int j = 0; j < bin.cols; j++) {
            if (edgeRow || j < b || j >= bin.cols - b) {
                total++;
                if (p[j]) fg++;
            }
        }
    }
    return total ? static_cast<double>(fg) / total : 0.0;
}
}  // namespace

int segmentImage(const cv::Mat& bgr, const SegmentOptions& opt, SegmentResult& out) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return -1;
    out = SegmentResult();
    out.resized = bgr;
    const int longer = std::max(bgr.cols, bgr.rows);
    if (opt.maxSide > 0 && longer > opt.maxSide) {
        const double scale = static_cast<double>(opt.maxSide) / longer;
        cv::resize(bgr, out.resized, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    if (toBinary(out.resized, out.binary, opt.channel)) return -1;

    // The background touches the image border: if most border pixels are "foreground" the objects
    // are the dark ones. (The original main() always inverted, vidDis/recognize never did.)
    out.inverted = opt.invert < 0 ? borderForeground(out.binary) > 0.5 : opt.invert != 0;
    if (out.inverted) cv::bitwise_not(out.binary, out.binary);

    if (opt.clean) {
        if (denoise(out.binary, out.cleaned)) return -1;
    } else {
        out.cleaned = out.binary.clone();
    }
    const int minArea = opt.minArea >= 0 ? opt.minArea : static_cast<int>(0.0035 * out.cleaned.total());
    const std::vector<cv::Mat> masks = regionize(out.cleaned, out.colored, opt.maxRegions, minArea);
    for (const cv::Mat& m : masks) {
        RegionInfo r;
        r.mask = m;
        r.features = featureBox(m, r.box);
        r.area = cv::countNonZero(m);
        if (!r.features.empty()) out.regions.push_back(std::move(r));
    }
    return 0;
}

cv::Mat annotate(const SegmentResult& result, const std::vector<std::string>& labels) {
    cv::Mat img = result.resized.clone();
    for (size_t i = 0; i < result.regions.size(); i++) {
        const std::string name = i < labels.size() ? labels[i] : "";
        drawBox(result.regions[i].box, result.regions[i].features, img, name, 2);
    }
    return img;
}
