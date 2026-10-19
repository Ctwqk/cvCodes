/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 *
 * Camera calibration with a chessboard. The camera matrix, distortion coefficients and the pattern size are
 * stored in an intrinsics file (default intcPara.txt).
 *
 * Interactive (camera):  calibrate [--source 0] [--pattern 9x6] [--out intcPara.txt]
 *     s  keep the current frame (the corners must be visible); from the fifth kept frame on the
 *        parameters are recomputed after every new frame
 *     q  quit; you are asked whether to save the parameters
 * Batch:  calibrate --images DIR | --source VIDEO  [--every N] [--max-views N] [--min-views N]
 *                   [--pattern 9x6] --out FILE
 *     uses every N-th frame (default 1) in which the board is found and writes FILE without asking.
 */
#include <cstdio>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "ar_utils.h"

namespace {

cv::Size parsePattern(const std::string& s) {
    int w = 0, h = 0;
    char x = 0;
    if (std::sscanf(s.c_str(), "%d%c%d", &w, &x, &h) != 3 || (x != 'x' && x != 'X') || w < 3 || h < 3) {
        throw std::invalid_argument("--pattern must look like 9x6");
    }
    return cv::Size(w, h);
}

void report(const p4::CalibrationResult& r, cv::Size size) {
    std::printf("calibrated from %d views of %dx%d images\n", r.views, size.width, size.height);
    std::printf("camera matrix:\n");
    for (int i = 0; i < 3; i++)
        std::printf("  %10.3f %10.3f %10.3f\n", r.cameraMatrix.at<double>(i, 0), r.cameraMatrix.at<double>(i, 1),
                    r.cameraMatrix.at<double>(i, 2));
    std::printf("distortion coefficients:");
    for (double d : r.distCoeffs) std::printf(" %.5f", d);
    std::printf("\nRMS reprojection error: %.4f px\n", r.rms);
}

int saveIntrinsics(const std::string& path, cv::Size pattern, const p4::CalibrationResult& r) {
    p4::Intrinsics in;
    in.pattern = pattern;
    in.cameraMatrix = r.cameraMatrix;
    in.distCoeffs = r.distCoeffs;
    std::string err;
    if (!p4::writeIntrinsics(path, in, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 2;
    }
    std::printf("intrinsic parameters written to %s\n", path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help")) {
            std::printf("usage: %s [--source 0|video] [--images DIR] [--pattern 9x6] [--every N] [--max-views N] [--min-views N] [--out FILE] [--headless]\n", argv[0]);
            return 0;
        }
        const cv::Size pattern = parsePattern(args.get("pattern", "9x6"));
        const std::string outPath = args.get("out", "intcPara.txt");
        const int minViews = args.getInt("min-views", 5);  // the original needed five frames

        // ---- batch mode over a folder or a video ----------------------------------------------------
        if (args.has("images") || (args.has("source") && cvc::Viewer::fromArgs(args).headless())) {
            std::vector<std::vector<cv::Point2f>> all;
            cv::Size imageSize;
            const int every = std::max(1, args.getInt("every", 1));
            const int maxViews = args.getInt("max-views", 1 << 30);
            int seen = 0, usable = 0;
            auto consume = [&](const cv::Mat& frame) {
                if (frame.empty()) return;
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                imageSize = gray.size();
                if (seen++ % every != 0 || static_cast<int>(all.size()) >= maxViews) return;
                std::vector<cv::Point2f> corners;
                if (p4::findBoardCorners(gray, pattern, corners)) {
                    all.push_back(corners);
                    usable++;
                }
            };
            if (args.has("images")) {
                for (const std::string& f : cvc::listImages(args.get("images"))) consume(cv::imread(f, cv::IMREAD_COLOR));
            } else {
                cv::VideoCapture cap = cvc::openSource(args.get("source"));
                cv::Mat frame;
                while (cap.read(frame)) consume(frame);
            }
            std::printf("%d frames examined, chessboard found in %d\n", seen, usable);
            p4::CalibrationResult r;
            if (static_cast<int>(all.size()) < minViews || !p4::calibrateFromCorners(imageSize, pattern, all, r)) {
                std::fprintf(stderr, "error: need at least %d usable views, have %zu\n", minViews, all.size());
                return 2;
            }
            report(r, imageSize);
            return saveIntrinsics(outPath, pattern, r);
        }

        // ---- interactive mode -------------------------------------------------------------------------
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        std::vector<std::vector<cv::Point2f>> kept;
        p4::CalibrationResult result;
        bool calibrated = false;
        cv::Size imageSize;
        cv::Mat frame, gray;
        for (;;) {
            const int c = cv::waitKey(10);
            if (c == 'q') break;
            cap >> frame;
            if (frame.empty()) break;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            imageSize = gray.size();
            std::vector<cv::Point2f> corners;
            const bool found = p4::findBoardCorners(gray, pattern, corners);
            if (found) cv::drawChessboardCorners(frame, pattern, corners, true);
            if (c == 's' && found) {
                kept.push_back(corners);
                std::printf("kept %zu frame(s)\n", kept.size());
                if (static_cast<int>(kept.size()) >= minViews && p4::calibrateFromCorners(imageSize, pattern, kept, result)) {
                    calibrated = true;
                    report(result, imageSize);
                }
            }
            cv::imshow("cor", frame);
        }
        if (calibrated) {
            std::printf("save intrinsic parameters to %s? [Y/N]\n", outPath.c_str());
            std::string answer;
            std::cin >> answer;
            if (!answer.empty() && (answer[0] == 'Y' || answer[0] == 'y')) return saveIntrinsics(outPath, pattern, result);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
