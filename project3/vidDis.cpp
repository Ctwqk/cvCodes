/*
 *  Author: taiwei cui
 *  Time: Feb 27 2024
 *
 * Live segmentation and recognition of objects from a camera or a video file.
 *
 *   vidDis [--source 0|video.mp4|frames/%04d.png] [--db FILE] [--classify] [--features shape|dnn]
 *          [--metric scaled|euclid|cosine] [--k N] [--max-frames N] [--out DIR]
 *
 * Without --classify the system is in learning mode: it only draws the boxes and features.
 * With --classify the nearest database entry is written next to every box.
 *
 * Keys: 1-6   number of regions to show (at most) / region to focus on, see 't'
 *       t     toggle the meaning of 1-6 (number of regions <-> focused region)
 *       s     (while focusing a region) label that region and append it to the database;
 *             the label is read from the terminal
 *       q     quit
 */
#include <cstdio>
#include <iostream>

#include <opencv2/highgui.hpp>

#include "app_io.h"
#include "csv_util.h"
#include "dbtools.h"
#include "regionfeatures.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "classify", "help"});
        if (args.has("help")) {
            std::printf("usage: %s [--source 0|file] [--db FILE] [--classify] [--features shape|dnn] [--metric M] [--k N] [--max-frames N] [--out DIR]\n", argv[0]);
            return 0;
        }
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/vidDis");
        const FeatureKind kind = parseFeatureKind(args.get("features", "shape"));
        const std::string dbPath = args.get("db", defaultDbPath(kind));
        ClassifierConfig cfg;
        cfg.kind = parseDistanceKind(args.get("metric", "scaled"));
        cfg.k = args.getInt("k", 1);
        const int maxFrames = args.getInt("max-frames", -1);

        cvc::FeatureTable db;
        std::vector<float> scale;
        const bool classifyMode = args.has("classify");
        if (classifyMode) {
            std::string err;
            if (!cvc::readFeatureTable(dbPath, db, &err)) {
                std::fprintf(stderr, "error: %s\n", err.c_str());
                return 2;
            }
        }
        cv::dnn::Net net;
        if (kind == FeatureKind::Dnn) net = cv::dnn::readNet(args.get("model", defaultModelPath()));

        int selectNum = kMaxRegions;  // how many regions to show
        int targetNum = 0;            // region in focus
        int toggle = 1;               // 1: keys 1-6 set the number of regions, -1: they choose the focused region
        cv::Mat frame;
        for (int n = 0; maxFrames < 0 || n < maxFrames; n++) {
            cap >> frame;
            if (frame.empty()) break;
            const int key = viewer.waitKey(10);
            bool label = false;
            if (key == 'q') break;
            if (toggle == -1 && key == 's') {
                label = true;
            } else if (key >= '1' && key <= '6') {
                if (toggle == 1) selectNum = key - '0';
                else targetNum = key - '1';
            } else if (key == 't') {
                toggle *= -1;
            }

            SegmentOptions opt;
            opt.maxSide = 0;  // keep the camera resolution
            opt.maxRegions = selectNum;
            SegmentResult seg;
            if (segmentImage(frame, opt, seg)) continue;

            std::vector<size_t> shown;
            if (toggle == -1) {
                if (!seg.regions.empty()) shown.push_back(std::min(static_cast<size_t>(targetNum), seg.regions.size() - 1));
            } else {
                for (size_t i = 0; i < seg.regions.size(); i++) shown.push_back(i);
            }
            cv::Mat view = seg.resized.clone();
            for (size_t i : shown) {
                const RegionInfo& r = seg.regions[i];
                const std::vector<float> v = regionFeature(r, kind, &net);
                if (label) {
                    std::string name;
                    std::cout << "please input the label of the current region: " << std::flush;
                    std::cin >> name;
                    std::string err;
                    if (!cvc::appendFeatureRow(dbPath, name, v, false, &err)) std::fprintf(stderr, "%s\n", err.c_str());
                }
                std::string name;
                if (classifyMode) {
                    if (cfg.kind == DistanceKind::ScaledEuclidean && scale.size() != v.size()) {
                        scale = columnStdDev(db, -1, v.size());
                    }
                    name = classify(v, db, cfg, scale).label;
                }
                drawBox(r.box, r.features, view, name);
                drawBox(r.box, r.features, seg.colored, name);
            }
            viewer.show("reg", seg.colored);
            viewer.show("video", view);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
