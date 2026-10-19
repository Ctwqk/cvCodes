/*
  Evaluate the exported MNIST network (OpenCV DNN, C++) on the MNIST test set.

  usage: mnistEval [--model mnist_model.onnx] [--data DIR] [--limit N] [--out DIR]
    --model   ONNX file (default project5/mnist_model.onnx in the repository)
    --data    folder with t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte (default project5/data/MNIST/raw)
    --limit   evaluate only the first N images
    --out     write confusion_matrix.txt, samples.png (first images) and errors.png (misclassified ones)
*/
#include <cstdio>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "idx.h"
#include "mnist.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        if (args.has("help")) {
            std::printf("usage: %s [--model FILE] [--data DIR] [--limit N] [--out DIR]\n", argv[0]);
            return 0;
        }
        const std::string model = args.get("model", cvc::repoPath("project5/mnist_model.onnx"));
        const std::string dir = args.get("data", cvc::repoPath("project5/data/MNIST/raw"));
        const size_t limit = static_cast<size_t>(std::max(0, args.getInt("limit", 0)));

        std::vector<cv::Mat> images;
        std::vector<uint8_t> labels;
        std::string err;
        if (!p5::readIdxImages(dir + "/t10k-images-idx3-ubyte", images, &err, limit) ||
            !p5::readIdxLabels(dir + "/t10k-labels-idx1-ubyte", labels, &err, limit)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        cv::dnn::Net net = cv::dnn::readNet(model);
        const p5::EvalResult r = p5::evaluate(net, images, labels);
        std::printf("%s on %d MNIST test images: %d correct, accuracy %.2f%%\n", model.c_str(), r.total, r.correct, 100 * r.accuracy());
        std::printf("confusion matrix (rows = true digit, columns = predicted):\n      ");
        for (int j = 0; j < 10; j++) std::printf("%5d", j);
        std::printf("\n");
        for (int i = 0; i < 10; i++) {
            std::printf("  %2d: ", i);
            for (int j = 0; j < 10; j++) std::printf("%5d", r.confusion[i][j]);
            std::printf("\n");
        }
        if (args.has("out")) {
            const std::string out = args.get("out");
            std::vector<size_t> first;
            for (size_t i = 0; i < std::min<size_t>(32, images.size()); i++) first.push_back(i);
            cvc::writeImage(out + "/samples.png", p5::predictionSheet(net, images, labels, first));
            std::vector<size_t> wrong(r.wrong.begin(), r.wrong.begin() + static_cast<long>(std::min<size_t>(32, r.wrong.size())));
            if (!wrong.empty()) cvc::writeImage(out + "/errors.png", p5::predictionSheet(net, images, labels, wrong));
            std::ofstream cm(out + "/confusion_matrix.txt");
            cm << "accuracy " << r.accuracy() << " (" << r.correct << "/" << r.total << ")\n";
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) cm << r.confusion[i][j] << (j < 9 ? " " : "\n");
            }
            std::printf("results written to %s\n", out.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
