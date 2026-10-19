#include "dnnEmbedding.h"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const char* const kEmbeddingLayer = "onnx_node!/fc1/Gemm";

int getEmbedding(const cv::Mat& src, cv::Mat& embedding, const cv::Rect& bboxIn, cv::dnn::Net& net, int debug) {
    const int ORNet_size = 128;
    if (src.empty() || src.type() != CV_8UC1) return -1;
    const cv::Rect bbox = bboxIn & cv::Rect(0, 0, src.cols, src.rows);
    if (bbox.area() <= 0) return -1;

    cv::Mat roiImg = src(bbox);
    // pad so that the object keeps its aspect ratio when resized to 128 x 128
    const int top = bbox.height > ORNet_size ? 10 : (ORNet_size - bbox.height) / 2 + 10;
    const int left = bbox.width > ORNet_size ? 10 : (ORNet_size - bbox.width) / 2 + 10;
    cv::Mat padImg;
    cv::copyMakeBorder(roiImg, padImg, top, top, left, left, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(ORNet_size, ORNet_size));

    cv::Mat blob;
    cv::dnn::blobFromImage(padImg,                         // input image (the sample passed `src` here)
                           blob,                           // output array
                           (1.0 / 255.0) / 0.5,            // scale factor
                           cv::Size(ORNet_size, ORNet_size),
                           128,                            // subtract mean prior to scaling
                           false,                          // swapRB (single channel anyway)
                           true,                           // crop
                           CV_32F);
    net.setInput(blob);
    embedding = net.forward(kEmbeddingLayer).clone();

    if (debug) {
        cv::imshow("pad image", padImg);
        std::cout << embedding << std::endl;
        cv::waitKey(0);
    }
    return 0;
}

std::vector<float> regionEmbedding(const cv::Mat& mask, const cv::RotatedRect& box, cv::dnn::Net& net) {
    cv::Mat emb;
    if (getEmbedding(mask, emb, box.boundingRect(), net, 0)) return {};
    cv::Mat flat = emb.reshape(1, 1);
    return std::vector<float>(flat.ptr<float>(0), flat.ptr<float>(0) + flat.cols);
}
