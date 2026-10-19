/*
  Run the project-3 segmentation pipeline on one image and show / save every stage:
  input -> threshold -> clean-up -> regions -> oriented boxes with shape features.

  usage: segment <image> [options]
    --out DIR         write <name>_1_input.png ... <name>_5_boxes.png into DIR
    --max-side N      shrink so that the longer side is at most N pixels (default 640, 0 = off)
    --channel N       threshold plane: 0 B, 1 G, 2 R, 3 grey, -1 automatic (default)
    --invert V        auto (default), 0 = bright objects on dark, 1 = dark objects on bright
    --regions N       keep at most N regions (1..6, default 6)
    --min-area N      ignore regions smaller than N pixels (default: 0.35 % of the image)
    --no-clean        skip the morphological clean-up
    --headless        never open windows (automatic when no display is available)
*/
#include <cstdio>
#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "pipeline.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "no-clean", "help"});
        if (args.has("help") || args.positional().empty()) {
            std::printf("usage: %s <image> [--out DIR] [--max-side N] [--channel N] [--invert auto|0|1] [--regions N] [--min-area N] [--no-clean] [--headless]\n", argv[0]);
            return args.has("help") ? 0 : 1;
        }
        const std::string path = args.positional()[0];
        const cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
        if (src.empty()) {
            std::fprintf(stderr, "error: unable to read image '%s'\n", path.c_str());
            return 2;
        }
        SegmentOptions opt;
        opt.maxSide = args.getInt("max-side", opt.maxSide);
        opt.channel = args.getInt("channel", opt.channel);
        const std::string inv = args.get("invert", "auto");
        opt.invert = inv == "auto" ? -1 : (inv == "0" ? 0 : (inv == "1" ? 1 : -2));
        if (opt.invert == -2) {
            std::fprintf(stderr, "error: --invert must be auto, 0 or 1\n");
            return 1;
        }
        opt.maxRegions = args.getInt("regions", opt.maxRegions);
        opt.minArea = args.getInt("min-area", opt.minArea);
        opt.clean = !args.has("no-clean");

        SegmentResult r;
        if (segmentImage(src, opt, r)) {
            std::fprintf(stderr, "error: segmentation failed\n");
            return 2;
        }
        std::printf("%s: %dx%d (resized %dx%d), polarity %s, %zu region(s)\n", path.c_str(), src.cols, src.rows,
                    r.resized.cols, r.resized.rows, r.inverted ? "dark objects" : "bright objects", r.regions.size());
        for (size_t i = 0; i < r.regions.size(); i++) {
            const RegionInfo& g = r.regions[i];
            std::printf("  region %zu: %6d px, box %.0fx%.0f at (%.0f,%.0f) angle %.1f, features fill=%.3f aspect=%.3f compact=%.3f\n",
                        i + 1, g.area, g.box.size.width, g.box.size.height, g.box.center.x, g.box.center.y,
                        g.box.angle, g.features[0], g.features[1], g.features[2]);
        }

        const cv::Mat boxes = annotate(r);
        if (args.has("out")) {
            const std::string stem = std::filesystem::path(path).stem().string();
            const std::string dir = args.get("out");
            cvc::writeImage(dir + "/" + stem + "_1_input.png", r.resized);
            cvc::writeImage(dir + "/" + stem + "_2_binary.png", r.binary);
            cvc::writeImage(dir + "/" + stem + "_3_cleaned.png", r.cleaned);
            cvc::writeImage(dir + "/" + stem + "_4_regions.png", r.colored);
            cvc::writeImage(dir + "/" + stem + "_5_boxes.png", boxes);
        }
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args);
        if (!viewer.headless()) {
            cv::imshow("src", r.resized);
            cv::imshow("bin", r.binary);
            cv::imshow("pur", r.cleaned);
            cv::imshow("reg", r.colored);
            cv::imshow("boxes", boxes);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
