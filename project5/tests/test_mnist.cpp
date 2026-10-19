#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "doctest.h"
#include "idx.h"
#include "mnist.h"

using namespace p5;
using cv::Mat;

namespace {
cv::dnn::Net& net() {
    static cv::dnn::Net n = cv::dnn::readNet(cvc::repoPath("project5/mnist_model.onnx"));
    return n;
}
struct Data {
    std::vector<cv::Mat> images;
    std::vector<uint8_t> labels;
};
const Data& testData() {
    static Data d = [] {
        Data x;
        const std::string dir = cvc::repoPath("project5/data/MNIST/raw");
        std::string err;
        const bool ok = readIdxImages(dir + "/t10k-images-idx3-ubyte", x.images, &err, 1000) &&
                        readIdxLabels(dir + "/t10k-labels-idx1-ubyte", x.labels, &err, 1000);
        REQUIRE_MESSAGE(ok, err);
        return x;
    }();
    return d;
}
}  // namespace

TEST_CASE("the exported network classifies the first 1000 MNIST test digits with >= 97 % accuracy") {
    const EvalResult r = evaluate(net(), testData().images, testData().labels);
    CHECK(r.total == 1000);
    CHECK(r.accuracy() >= 0.97);
    int sum = 0;
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++) sum += r.confusion[i][j];
    CHECK(sum == r.total);
    CHECK(r.correct + static_cast<int>(r.wrong.size()) == r.total);
}

TEST_CASE("predictDigit returns a probability distribution and the argmax") {
    const Prediction p = predictDigit(net(), testData().images[0]);  // a 7
    REQUIRE(p.probabilities.size() == 10);
    float sum = 0;
    for (float v : p.probabilities) {
        CHECK(v >= 0.f);
        sum += v;
    }
    CHECK(sum == doctest::Approx(1.f));
    CHECK(p.digit == 7);
    CHECK(p.probability == doctest::Approx(p.probabilities[7]));
    CHECK(p.probability > 0.9f);
}

TEST_CASE("predictDigit rejects pictures of the wrong size or type") {
    CHECK(predictDigit(net(), Mat()).digit == -1);
    CHECK(predictDigit(net(), Mat(30, 30, CV_8UC1)).digit == -1);
    CHECK(predictDigit(net(), Mat(28, 28, CV_8UC3)).digit == -1);
}

TEST_CASE("makeBlob matches the training normalisation (x/255 - 0.5)/0.5") {
    Mat g(28, 28, CV_8UC1, cv::Scalar(0));
    g.at<uchar>(0, 1) = 255;
    g.at<uchar>(0, 2) = 51;
    const Mat blob = makeBlob(g);
    REQUIRE(blob.dims == 4);
    CHECK(blob.size[1] == 1);
    CHECK(blob.size[2] == 28);
    const float* px = blob.ptr<float>();  // row-major 28x28
    CHECK(px[0] == doctest::Approx(-1.f));
    CHECK(px[1] == doctest::Approx(1.f));
    CHECK(px[2] == doctest::Approx((51.0 / 255 - 0.5) / 0.5));
}

TEST_CASE("evaluate skips out-of-range labels and handles mismatched lengths") {
    std::vector<Mat> imgs(testData().images.begin(), testData().images.begin() + 3);
    const EvalResult r = evaluate(net(), imgs, {7, 12});  // only two labels, one of them invalid
    CHECK(r.total == 1);
    CHECK(evaluate(net(), {}, {}).accuracy() == 0.0);
}

TEST_CASE("preprocessDigit: dark ink on light paper becomes a centred white 20x20-box digit") {
    Mat paper(480, 640, CV_8UC3, cv::Scalar(215, 220, 225));
    cv::putText(paper, "3", cv::Point(420, 330), cv::FONT_HERSHEY_SIMPLEX, 6.0, cv::Scalar(40, 40, 40), 14);
    cv::Rect box;
    const Mat d = preprocessDigit(paper, &box);
    REQUIRE_FALSE(d.empty());
    CHECK(d.size() == cv::Size(28, 28));
    CHECK(d.type() == CV_8UC1);
    CHECK(box.x > 380);  // the digit blob was found where it was drawn
    // white on black, ink is a minority of the pixels
    CHECK(cv::mean(d)[0] < 100);
    CHECK(cv::mean(d)[0] > 15);
    CHECK(d.at<uchar>(0, 0) == 0);
    // centre of mass at the image centre
    const cv::Moments m = cv::moments(d);
    CHECK(m.m10 / m.m00 == doctest::Approx(13.5).epsilon(0.1));
    CHECK(m.m01 / m.m00 == doctest::Approx(13.5).epsilon(0.1));
    // the digit fits the 20x20 box (4 pixels of margin all around)
    Mat cols, rows;
    cv::reduce(d, cols, 0, cv::REDUCE_MAX);
    cv::reduce(d, rows, 1, cv::REDUCE_MAX);
    CHECK(cols.at<uchar>(0, 0) == 0);
    CHECK(cols.at<uchar>(0, 27) == 0);
    CHECK(rows.at<uchar>(0, 0) == 0);
    CHECK(rows.at<uchar>(27, 0) == 0);
}

TEST_CASE("preprocessDigit: light ink on a dark background gives the same kind of picture; blank pictures give nothing") {
    Mat dark(300, 300, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::putText(dark, "8", cv::Point(80, 240), cv::FONT_HERSHEY_SIMPLEX, 6.0, cv::Scalar(230, 230, 230), 12);
    const Mat d = preprocessDigit(dark);
    REQUIRE_FALSE(d.empty());
    CHECK(cv::mean(d)[0] < 100);
    CHECK(preprocessDigit(Mat(200, 200, CV_8UC3, cv::Scalar(128, 128, 128))).empty() == true);
    CHECK(preprocessDigit(Mat()).empty());
    Mat gray16(10, 10, CV_16UC1, cv::Scalar(5));
    CHECK(preprocessDigit(gray16).empty());
}

TEST_CASE("preprocessDigit output of a real MNIST digit is classified correctly (re-centred, enlarged copy)") {
    // enlarge an MNIST test digit (white on black) into a camera-like frame with dark ink on paper
    const Data& d = testData();
    int correct = 0, tried = 0;
    for (size_t i = 0; i < 40; i++) {
        Mat big, frame;
        cv::resize(d.images[i], big, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);
        cv::bitwise_not(big, big);  // dark ink on light paper
        cv::copyMakeBorder(big, frame, 60, 60, 120, 120, cv::BORDER_CONSTANT, cv::Scalar(235));
        const Mat out = preprocessDigit(frame);
        if (out.empty()) continue;
        tried++;
        if (predictDigit(net(), out).digit == d.labels[i]) correct++;
    }
    CHECK(tried >= 38);
    CHECK(correct >= tried - 4);  // the full pipeline loses a few digits to thresholding, not many
}

TEST_CASE("predictionSheet lays out one captioned cell per sample") {
    const Data& d = testData();
    const Mat sheet = predictionSheet(net(), d.images, d.labels, {0, 1, 2, 3, 4, 5}, 3, 2);
    CHECK(sheet.cols == 3 * 56);
    CHECK(sheet.rows == 2 * (56 + 16));
    CHECK(sheet.type() == CV_8UC3);
}
