/*
  Bruce A. Maxwell
  CS 5330 Spring 2024
  Project 3

  Reads a 2DOR network and an image and applies the image to the network to generate an embedding.
  Also prints out the model layers. The fc1/Gemm layer is used as the embedding.

  usage: dnnexample [--model ONNX] <thresholded image> [--layers]
         (default model: project3/NetWorkThings/or2d-normmodel-007.onnx)
  The image should be a thresholded image with one object in it.
*/
#include <cstdio>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "dbtools.h"
#include "dnnEmbedding.h"

int main(int argc, char* argv[]) {
    try {
        const cvc::Args args(argc, argv, {"layers", "help"});
        if (args.has("help") || args.positional().empty()) {
            std::printf("usage: %s [--model ONNX] <thresholded image> [--layers]\n", argv[0]);
            return args.has("help") ? 0 : 1;
        }
        cv::dnn::Net net = cv::dnn::readNet(args.get("model", defaultModelPath()));
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
        cv::Mat embedding;
        // the box that contains the object to be recognised: here the whole image
        if (getEmbedding(src, embedding, cv::Rect(0, 0, src.cols, src.rows), net, 0)) return 2;
        std::printf("embedding (%d values):\n", embedding.cols);
        for (int i = 0; i < embedding.cols; i++) std::printf("%s%.4f", i ? " " : "", static_cast<double>(embedding.at<float>(0, i)));
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
