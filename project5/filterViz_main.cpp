/*
  Apply the ten 5x5 filters of the network's first convolution layer to a digit image.

  usage: filterViz [--filters filters.npy] [--image image.npy] [--out DIR]
    --filters  (10, 5, 5) float array exported by export_filters.py (default project5/filters.npy)
    --image    (28, 28) float array in [0, 1] (default project5/image.npy, the first MNIST training image)
    --out      write ansImage<i>.npy (the responses, as the original program did) and filter_responses.png
*/
#include <cstdio>
#include <filesystem>

#include "app_io.h"
#include "filterviz.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        if (args.has("help")) {
            std::printf("usage: %s [--filters FILE.npy] [--image FILE.npy] [--out DIR]\n", argv[0]);
            return 0;
        }
        p5::NpyArray filters, image;
        std::string err;
        if (!p5::readNpy(args.get("filters", cvc::repoPath("project5/filters.npy")), filters, &err) ||
            !p5::readNpy(args.get("image", cvc::repoPath("project5/image.npy")), image, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        std::vector<cv::Mat> responses, kernels;
        cv::Mat img;
        if (!p5::applyFilters(filters, image, responses, &kernels, &img)) {
            std::fprintf(stderr, "error: expected filters of shape (N, h, w) and an image of shape (rows, cols)\n");
            return 2;
        }
        std::printf("%zu filters of %zux%zu applied to a %zux%zu image\n", responses.size(), filters.shape[1], filters.shape[2],
                    image.shape[0], image.shape[1]);
        for (size_t i = 0; i < responses.size(); i++) {
            double lo = 0, hi = 0;
            cv::minMaxLoc(responses[i], &lo, &hi);
            std::printf("  filter %2zu: response range [%.3f, %.3f]\n", i + 1, lo, hi);
        }
        if (args.has("out")) {
            const std::string dir = args.get("out");
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            for (size_t i = 0; i < responses.size(); i++) {
                const cv::Mat flat = responses[i].reshape(1, 1);
                const std::vector<float> values(flat.ptr<float>(0), flat.ptr<float>(0) + flat.cols);
                if (!p5::writeNpy(dir + "/ansImage" + std::to_string(i + 1) + ".npy", {static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)}, values, &err)) {
                    std::fprintf(stderr, "error: %s\n", err.c_str());
                    return 2;
                }
            }
            cvc::writeImage(dir + "/filter_responses.png", p5::filterSheet(img, kernels, responses));
            std::printf("results written to %s\n", dir.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
