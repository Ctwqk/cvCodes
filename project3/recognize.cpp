/*
  Classify the objects in a photo with the nearest-neighbour database (or add them to it).

  usage: recognize <image> [options]
    --db FILE        feature database (default project3/trainset/<features>_features.csv)
    --features K     shape (default) or dnn
    --metric M       scaled (default), euclid or cosine
    --k N            k-nearest-neighbour vote (default 1)
    --model ONNX     embedding network for --features dnn
    --regions N      classify at most N regions (default 6)
    --out DIR        save the annotated picture to DIR/<name>_recognized.png
    --hold-out       ignore database rows that came from this very image (honest test of a training photo)
    --learn LABEL    do not classify: append the largest region to the database as LABEL:<image>
    --headless       never open windows
*/
#include <cstdio>
#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "csv_util.h"
#include "dbtools.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help", "hold-out"});
        if (args.has("help") || args.positional().empty()) {
            std::printf("usage: %s <image> [--db FILE] [--features shape|dnn] [--metric scaled|euclid|cosine] [--k N] [--model ONNX] [--regions N] [--out DIR] [--hold-out] [--learn LABEL] [--headless]\n", argv[0]);
            return args.has("help") ? 0 : 1;
        }
        const std::string path = args.positional()[0];
        const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::fprintf(stderr, "error: unable to read image '%s'\n", path.c_str());
            return 2;
        }
        const FeatureKind kind = parseFeatureKind(args.get("features", "shape"));
        const std::string dbPath = args.get("db", defaultDbPath(kind));
        ClassifierConfig cfg;
        cfg.kind = parseDistanceKind(args.get("metric", "scaled"));
        cfg.k = args.getInt("k", 1);

        SegmentOptions opt;
        opt.maxRegions = args.has("learn") ? 1 : args.getInt("regions", kMaxRegions);
        SegmentResult seg;
        if (segmentImage(img, opt, seg) || seg.regions.empty()) {
            std::fprintf(stderr, "no object found in %s\n", path.c_str());
            return 3;
        }

        cv::dnn::Net net;
        if (kind == FeatureKind::Dnn) net = cv::dnn::readNet(args.get("model", defaultModelPath()));

        if (args.has("learn")) {
            const std::vector<float> v = regionFeature(seg.regions[0], kind, &net);
            std::string err;
            if (!cvc::appendFeatureRow(dbPath, args.get("learn") + ":" + cvc::baseName(path), v, false, &err)) {
                std::fprintf(stderr, "error: %s\n", err.c_str());
                return 2;
            }
            std::printf("added '%s' (%zu values) to %s\n", args.get("learn").c_str(), v.size(), dbPath.c_str());
            return 0;
        }

        cvc::FeatureTable db;
        std::string err;
        if (!cvc::readFeatureTable(dbPath, db, &err)) {
            std::fprintf(stderr, "error: %s\n(build it with buildDb)\n", err.c_str());
            return 2;
        }
        if (args.has("hold-out")) {
            cvc::FeatureTable kept;
            const std::string source = cvc::baseName(path);
            for (size_t i = 0; i < db.size(); i++) {
                const size_t colon = db.names[i].find(':');
                if (colon != std::string::npos && db.names[i].substr(colon + 1) == source) continue;
                kept.names.push_back(db.names[i]);
                kept.data.push_back(db.data[i]);
            }
            db = std::move(kept);
        }
        const std::vector<float> scale = columnStdDev(db);
        std::vector<std::string> labels;
        for (size_t i = 0; i < seg.regions.size(); i++) {
            const std::vector<float> v = regionFeature(seg.regions[i], kind, &net);
            const Classification c = classify(v, db, cfg, scale);
            labels.push_back(c.label.empty() ? "?" : c.label);
            std::printf("region %zu: %-8s (distance %.3f, nearest %s)\n", i + 1, labels.back().c_str(),
                        static_cast<double>(c.distance), c.neighbours.empty() ? "-" : db.names[c.neighbours[0]].c_str());
        }

        const cv::Mat shown = annotate(seg, labels);
        if (args.has("out")) {
            const std::string stem = std::filesystem::path(path).stem().string();
            cvc::writeImage(args.get("out") + "/" + stem + "_recognized.png", shown);
        }
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args);
        if (!viewer.headless()) {
            cv::imshow("recognized", shown);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
