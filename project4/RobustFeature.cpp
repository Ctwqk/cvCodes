/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 *
 * Harris corners as robust features: the Harris response is normalised to [0, 255] and every point above the
 * threshold is marked.
 *
 *   RobustFeature [--source 0|video|pattern|image] [--threshold 160] [--all-pixels] [--out DIR]
 *                 [--max-frames N] [--save-every N] [--headless]
 *
 * By default only local maxima are marked (one ring per corner); --all-pixels marks every pixel above the
 * threshold as the original did.
 */
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "ar_utils.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help", "all-pixels"});
        if (args.has("help")) {
            std::printf("usage: %s [--source 0|video|pattern|image] [--threshold T] [--all-pixels] [--out DIR] [--max-frames N] [--save-every N] [--headless]\n", argv[0]);
            return 0;
        }
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/RobustFeature");
        const double threshold = args.getDouble("threshold", 160);  // user-setting threshold
        const bool nms = !args.has("all-pixels");
        const int maxFrames = args.getInt("max-frames", -1);

        cv::Mat frame, gray;
        size_t total = 0;
        int frames = 0;
        for (int n = 0; maxFrames < 0 || n < maxFrames; n++) {
            if (viewer.waitKey(10) == 'q') break;
            cap >> frame;
            if (frame.empty()) break;
            frames++;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            const std::vector<cv::Point2f> corners = p4::harrisCorners(gray, threshold, nms);
            total += corners.size();
            for (const cv::Point2f& p : corners) cv::circle(frame, p, 5, cv::Scalar(0, 0, 255));
            viewer.show("vid", frame);
        }
        std::printf("%d frames, %.1f corners per frame above threshold %.0f\n", frames, frames ? static_cast<double>(total) / frames : 0.0, threshold);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
