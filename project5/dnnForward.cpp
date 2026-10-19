/*
  Bruce A. Maxwell
  CS 5330 Spring 2024

  Reads a network and an image and prints the network output for it (and, with --layers, the layer names).

  usage: dnnForward [--model mnist_model.onnx] <image> [--layers]
*/
#include <cstdio>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "dnnEmbedding.h"

int main(int argc, char* argv[]) {
    try {
        const cvc::Args args(argc, argv, {"layers", "help"});
        if (args.has("help") || args.positional().empty()) {
            std::printf("usage: %s [--model FILE] <image> [--layers]\n", argv[0]);
            return args.has("help") ? 0 : 1;
        }
        cv::dnn::Net net = cv::dnn::readNet(args.get("model", cvc::repoPath("project5/mnist_model.onnx")));
        std::printf("Network read successfully\n");
        if (args.has("layers")) {
            const std::vector<cv::String> names = net.getLayerNames();
            for (size_t i = 0; i < names.size(); i++) std::printf("Layer %zu: '%s'\n", i, names[i].c_str());
        }
        cv::Mat src = cv::imread(args.positional()[0]);
        if (src.empty()) {
            std::fprintf(stderr, "error: unable to read image\n");
            return 2;
        }
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::Mat out;
        if (getEmbedding(src, out, cv::Rect(0, 0, src.cols, src.rows), net, 0)) return 2;
        for (int i = 0; i < out.cols; i++) std::printf("%s%.4f", i ? " " : "", static_cast<double>(out.at<float>(0, i)));
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
