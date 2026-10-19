/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Simple example of face detection using a Haar cascade
  (adapted: any video source, optional headless output).

  usage: showFaces [--source 0|video.mp4|image.jpg] [--out DIR] [--max-frames N]
*/
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "faceDetect.h"

int main(int argc, char* argv[]) {
    try {
        const cvc::Args args(argc, argv, {"headless"});
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/showFaces");
        const int maxFrames = args.getInt("max-frames", -1);

        cv::Mat frame, grey;
        std::vector<cv::Rect> faces;
        for (int f = 0; maxFrames < 0 || f < maxFrames; f++) {
            cap >> frame;
            if (frame.empty()) break;
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
            if (detectFaces(grey, faces)) return 2;
            drawBoxes(frame, faces);
            viewer.show("Video", frame);
            if (viewer.waitKey(10) == 'q') break;
        }
        std::printf("Terminating\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
