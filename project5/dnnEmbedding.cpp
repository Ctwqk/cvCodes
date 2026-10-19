#include "dnnEmbedding.h"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int getEmbedding(const cv::Mat& src, cv::Mat& embedding, const cv::Rect& bboxIn, cv::dnn::Net& net, int debug) {
    const int ORNet_size = 28;
    if (src.empty() || src.type() != CV_8UC1) return -1;
    const cv::Rect bbox = bboxIn & cv::Rect(0, 0, src.cols, src.rows);
    if (bbox.area() <= 0) return -1;
    const cv::Mat roiImg = src(bbox);
    const int top = bbox.height > ORNet_size ? 10 : (ORNet_size - bbox.height) / 2 + 10;
    const int left = bbox.width > ORNet_size ? 10 : (ORNet_size - bbox.width) / 2 + 10;
    cv::Mat padImg;
    cv::copyMakeBorder(roiImg, padImg, top, top, left, left, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(ORNet_size, ORNet_size));

    // the sample fed `src` (the whole picture) here instead of the padded region it had just built
    const cv::Mat blob = cv::dnn::blobFromImage(padImg, (1.0 / 255.0) / 0.5, cv::Size(ORNet_size, ORNet_size), 127.5, false, true, CV_32F);
    net.setInput(blob);
    embedding = net.forward().clone();

    if (debug) {
        cv::imshow("pad image", padImg);
        std::cout << embedding << std::endl;
        cv::waitKey(0);
    }
    return 0;
}
