/*
  Build the labelled feature database from a folder of training photos.

  usage: buildDb [--images DIR] [--features shape|dnn] [--db FILE] [--model ONNX] [options]
    --images DIR     training photos (default project3/trainset); the class is the file name without
                     extension and trailing digits: ball1.jpg -> "ball", spoon.jpg -> "spoon"
    --features K     shape (3 values, default) or dnn (50-value embedding)
    --db FILE        output CSV (default project3/trainset/<kind>_features.csv)
    --max-side N     see `segment` (default 640)
    --regions N      regions kept per photo whose name is plural (default 3); single-object photos keep the largest region
  Rows are "<label>:<image>,<values...>".
*/
#include <cstdio>
#include <map>

#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "csv_util.h"
#include "dbtools.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        if (args.has("help")) {
            std::printf("usage: %s [--images DIR] [--features shape|dnn] [--db FILE] [--model ONNX] [--max-side N] [--regions N]\n", argv[0]);
            return 0;
        }
        const FeatureKind kind = parseFeatureKind(args.get("features", "shape"));
        const std::string dir = args.get("images", cvc::repoPath("project3/trainset"));
        const std::string dbPath = args.get("db", defaultDbPath(kind));
        SegmentOptions opt;
        opt.maxSide = args.getInt("max-side", opt.maxSide);
        const int pluralRegions = args.getInt("regions", 3);

        cv::dnn::Net net;
        if (kind == FeatureKind::Dnn) net = cv::dnn::readNet(args.get("model", defaultModelPath()));

        const std::vector<std::string> files = cvc::listImages(dir);
        if (files.empty()) {
            std::fprintf(stderr, "error: no images in %s\n", dir.c_str());
            return 2;
        }
        int rows = 0;
        std::map<std::string, int> perClass;
        for (const std::string& f : files) {
            cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
            if (img.empty()) continue;
            std::string label = labelFromFilename(f);
            const bool plural = label.size() > 1 && label.back() == 's';
            if (plural) label.pop_back();  // "balls" photos show several objects of class "ball"
            SegmentOptions o = opt;
            o.maxRegions = plural ? pluralRegions : 1;
            SegmentResult r;
            if (segmentImage(img, o, r) || r.regions.empty()) {
                std::fprintf(stderr, "warning: no region found in %s\n", f.c_str());
                continue;
            }
            for (const RegionInfo& region : r.regions) {
                const std::vector<float> v = regionFeature(region, kind, &net);
                if (v.empty()) continue;
                std::string err;
                if (!cvc::appendFeatureRow(dbPath, label + ":" + cvc::baseName(f), v, rows == 0, &err)) {
                    std::fprintf(stderr, "error: %s\n", err.c_str());
                    return 2;
                }
                rows++;
                perClass[label]++;
            }
            std::printf("%-26s -> %s (%zu region%s)\n", cvc::baseName(f).c_str(), label.c_str(), r.regions.size(),
                        r.regions.size() == 1 ? "" : "s");
        }
        std::printf("%d rows written to %s\n", rows, dbPath.c_str());
        for (const auto& [label, n] : perClass) std::printf("  %-8s %d\n", label.c_str(), n);
        return rows ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
