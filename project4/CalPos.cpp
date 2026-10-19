/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 *
 * Detects the chessboard in every frame, estimates the camera pose with solvePnP and draws a virtual
 * 5-point star and a set of 3-D axes floating above the board.
 *
 *   CalPos [--source 0|video|frames/%04d.png] [--calib intcPara.txt] [--out DIR] [--max-frames N]
 *          [--save-every N] [--headless]
 *
 * --calib defaults to project4/intcPara.txt (the original webcam calibration) in the repository.
 * In headless mode every N-th processed frame is written to DIR/vid_<n>.png.
 */
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "ar_utils.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help")) {
            std::printf("usage: %s [--source 0|video|pattern] [--calib FILE] [--out DIR] [--max-frames N] [--save-every N] [--headless]\n", argv[0]);
            return 0;
        }
        p4::Intrinsics cam;
        std::string err;
        if (!p4::readIntrinsics(args.get("calib", cvc::repoPath("project4/intcPara.txt")), cam, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/CalPos");
        const int maxFrames = args.getInt("max-frames", -1);

        int frames = 0, detected = 0;
        double errSum = 0;
        cv::Mat frame, gray, rvec, tvec;
        for (int n = 0; maxFrames < 0 || n < maxFrames; n++) {
            if (viewer.waitKey(3) == 'q') break;
            cap >> frame;
            if (frame.empty()) break;
            frames++;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> corners;
            if (p4::findBoardCorners(gray, cam.pattern, corners) && p4::estimatePose(cam, corners, rvec, tvec)) {
                detected++;
                errSum += p4::reprojectionError(cam, corners, rvec, tvec);
                for (const cv::Point2f& p : corners) cv::circle(frame, p, 1, cv::Scalar(0, 0, 255), -1);
                p4::drawPoseText(frame, rvec, tvec);
                p4::drawAxes(frame, cam, rvec, tvec);
                p4::drawStar(frame, cam, rvec, tvec);
            }
            viewer.show("vid", frame);
        }
        std::printf("%d frames, board found in %d", frames, detected);
        if (detected) std::printf(", mean corner reprojection error %.3f px", errSum / detected);
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
