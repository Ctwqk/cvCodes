/*
  Live digit recognition: every frame of a camera / video / image is turned into an MNIST-style 28x28 image
  and classified by the network.

  usage: mnistLive [--source 0|video|image] [--model mnist_model.onnx] [--out DIR] [--max-frames N] [--headless]
  Shows the camera frame with the recognised digit and the 28x28 network input. Headless runs save these
  pictures to DIR/live_<n>.png and print one line per frame. 'q' quits.

  (The original program fed the whole 640x480 frame to the network and printed a log-probability as
  "likelyhood"; now the digit is cropped and centred first and the probability is reported.)
*/
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "mnist.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help")) {
            std::printf("usage: %s [--source 0|video|image] [--model FILE] [--out DIR] [--max-frames N] [--save-every N] [--headless]\n", argv[0]);
            return 0;
        }
        cv::dnn::Net net = cv::dnn::readNet(args.get("model", cvc::repoPath("project5/mnist_model.onnx")));
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/mnistLive");
        const int maxFrames = args.getInt("max-frames", -1);

        cv::Mat frame;
        for (int n = 0; maxFrames < 0 || n < maxFrames; n++) {
            if (viewer.waitKey(3) == 'q') break;
            cap >> frame;
            if (frame.empty()) break;
            cv::Rect box;
            const cv::Mat digit = p5::preprocessDigit(frame, &box);
            cv::Mat shown = frame.clone();
            if (digit.empty()) {
                std::printf("frame %d: no digit found\n", n);
                cv::putText(shown, "no digit", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
            } else {
                const p5::Prediction p = p5::predictDigit(net, digit);
                std::printf("frame %d: digit %d, probability %.3f\n", n, p.digit, static_cast<double>(p.probability));
                cv::rectangle(shown, box, cv::Scalar(0, 200, 0), 2);
                char text[48];
                std::snprintf(text, sizeof text, "%d (%.0f%%)", p.digit, 100.0 * static_cast<double>(p.probability));
                cv::putText(shown, text, cv::Point(10, 34), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(0, 160, 0), 3);
                cv::Mat input;
                cv::resize(digit, input, cv::Size(112, 112), 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(input, input, cv::COLOR_GRAY2BGR);
                if (shown.cols > 120 && shown.rows > 120) input.copyTo(shown(cv::Rect(shown.cols - 120, 8, 112, 112)));
            }
            viewer.show("live", shown);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
