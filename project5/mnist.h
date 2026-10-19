// MNIST digit recognition with the exported ONNX network (project 5).
#ifndef MNIST_H
#define MNIST_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace p5 {

struct Prediction {
    int digit = -1;
    float probability = 0.f;            // probability of `digit` (softmax of the log-probabilities the net outputs)
    std::vector<float> probabilities;   // 10 values
};

// Blob exactly as in training: (pixel / 255 - 0.5) / 0.5, shape 1x1x28x28.
cv::Mat makeBlob(const cv::Mat& gray28);

// Classify a 28x28 CV_8UC1 image (white digit on black, like MNIST). Digit -1 for wrong input.
Prediction predictDigit(cv::dnn::Net& net, const cv::Mat& gray28);

struct EvalResult {
    int total = 0;
    int correct = 0;
    int confusion[10][10] = {};         // confusion[truth][predicted]
    std::vector<size_t> wrong;          // indices of misclassified samples
    double accuracy() const { return total ? static_cast<double>(correct) / total : 0.0; }
};
EvalResult evaluate(cv::dnn::Net& net, const std::vector<cv::Mat>& images, const std::vector<uint8_t>& labels);

// Turn a camera frame / photo of a handwritten digit into an MNIST-style 28x28 image: threshold (Otsu,
// ink becomes white, polarity is detected automatically), keep the largest blob, scale it to fit a 20x20 box
// and centre it by its centre of mass in 28x28. Returns an empty Mat when no digit is found.
cv::Mat preprocessDigit(const cv::Mat& bgrOrGray, cv::Rect* digitBox = nullptr);

// A contact sheet of (28x28) samples with captions "truth > predicted", wrong ones in red.
cv::Mat predictionSheet(cv::dnn::Net& net, const std::vector<cv::Mat>& images, const std::vector<uint8_t>& labels,
                        const std::vector<size_t>& indices, int columns = 8, int scale = 3);

}  // namespace p5

#endif
