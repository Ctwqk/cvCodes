/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 *
 * Extension: finds every chessboard in the frame (several can be visible) and pastes a picture over each of
 * them so the target no longer looks like a chessboard, with the pose axes drawn on top.
 *
 *   extension [--source 0|video|pattern] [--calib FILE] [--replace IMAGE] [--out DIR] [--max-frames N]
 *             [--save-every N] [--headless]
 *
 * Defaults: --calib project4/intcPara.txt, --replace project4/smile.jpeg (inside the repository).
 */
#include <cstdio>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "ar_utils.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help")) {
            std::printf("usage: %s [--source 0|video|pattern] [--calib FILE] [--replace IMAGE] [--out DIR] [--max-frames N] [--save-every N] [--headless]\n", argv[0]);
            return 0;
        }
        const cv::Mat replacement = cv::imread(args.get("replace", cvc::repoPath("project4/smile.jpeg")));
        if (replacement.empty()) {
            std::fprintf(stderr, "error: failed to read the replacement image\n");
            return 2;
        }
        p4::Intrinsics cam;
        std::string err;
        if (!p4::readIntrinsics(args.get("calib", cvc::repoPath("project4/intcPara.txt")), cam, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/extension");
        const int maxFrames = args.getInt("max-frames", -1);

        int frames = 0, boards = 0;
        cv::Mat frame, gray, rvec, tvec;
        for (int n = 0; maxFrames < 0 || n < maxFrames; n++) {
            if (viewer.waitKey(3) == 'q') break;
            cap >> frame;
            if (frame.empty()) break;
            frames++;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            // keep looking while boards are found; every board is blacked out in the grey image once handled
            for (int found = 0; found < 8; found++) {
                std::vector<cv::Point2f> corners;
                if (!p4::findBoardCorners(gray, cam.pattern, corners)) break;
                boards++;
                const bool posed = p4::estimatePose(cam, corners, rvec, tvec);
                if (!p4::replaceBoard(frame, replacement, corners, cam.pattern, &gray)) break;
                if (posed) p4::drawAxes(frame, cam, rvec, tvec);
            }
            viewer.show("vid", frame);
        }
        std::printf("%d frames, %d board(s) replaced\n", frames, boards);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
