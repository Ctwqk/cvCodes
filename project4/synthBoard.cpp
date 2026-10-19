/*
 * Renders synthetic camera views of the printed chessboard (checkerboard.png) so that calibration and the AR
 * apps can run without a camera, and so that the recovered parameters can be compared with known truth.
 *
 *   synthBoard --out DIR [--mode calibration|video] [--frames N] [--seed S] [--board IMAGE]
 *              [--width 640] [--height 480] [--fx 520] [--k1 -0.18] [--k2 0.05]
 *
 *   calibration  N independent random poses -> DIR/calib_0000.png ...  (default N = 20)
 *   video        a smooth fly-by of N frames -> DIR/frame_0000.png ... (default N = 60)
 * DIR/ground_truth.txt holds the true camera in the intrinsics file format.
 */
#include <cstdio>
#include <filesystem>

#include "app_io.h"
#include "ar_utils.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        if (args.has("help") || !args.has("out")) {
            std::printf("usage: %s --out DIR [--mode calibration|video] [--frames N] [--seed S] [--board IMAGE] [--width W] [--height H] [--fx F] [--k1 K] [--k2 K]\n", argv[0]);
            return args.has("help") ? 0 : 1;
        }
        const std::string mode = args.get("mode", "calibration");
        if (mode != "calibration" && mode != "video") {
            std::fprintf(stderr, "error: --mode must be calibration or video\n");
            return 1;
        }
        const std::string dir = args.get("out");
        const int frames = args.getInt("frames", mode == "video" ? 60 : 20);
        const unsigned seed = static_cast<unsigned>(args.getInt("seed", 1));
        const cv::Size size(args.getInt("width", 640), args.getInt("height", 480));

        p4::BoardTemplate board;
        std::string err;
        if (!p4::loadBoardTemplate(args.get("board", cvc::repoPath("project4/checkerboard.png")), cv::Size(9, 6), board, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        p4::Intrinsics truth;
        const double fx = args.getDouble("fx", 520);
        truth.cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, size.width / 2.0, 0, fx, size.height / 2.0, 0, 0, 1);
        truth.distCoeffs = {args.getDouble("k1", -0.18), args.getDouble("k2", 0.05), 0.0005, -0.0003, 0.0};
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (!p4::writeIntrinsics(dir + "/ground_truth.txt", truth, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }

        for (int i = 0; i < frames; i++) {
            cv::Mat rvec, tvec;
            if (mode == "video") p4::trajectoryPose(frames > 1 ? static_cast<double>(i) / (frames - 1) : 0.0, rvec, tvec);
            else p4::samplePose(i, frames, seed, rvec, tvec);
            const cv::Mat view = p4::renderBoardView(board, truth, rvec, tvec, size, seed * 1000u + static_cast<unsigned>(i));
            char name[64];
            std::snprintf(name, sizeof name, mode == "video" ? "/frame_%04d.png" : "/calib_%04d.png", i);
            if (!cvc::writeImage(dir + name, view)) {
                std::fprintf(stderr, "error: cannot write %s%s\n", dir.c_str(), name);
                return 2;
            }
        }
        std::printf("%d %s views written to %s (fx=fy=%.1f, k1=%.3f, k2=%.3f)\n", frames, mode.c_str(), dir.c_str(), fx,
                    truth.distCoeffs[0], truth.distCoeffs[1]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
