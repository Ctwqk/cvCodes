/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Execute k-means on the colors of an image and re-render the image with the
  K colors chosen by the algorithm.

  Original course skeleton, extended in the portfolio upgrade with a proper
  command line, seeded sampling, a palette report and file output (so it can
  run without a display).
*/
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <random>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "colorquant.h"
#include "kmeans.h"

static void usage(const char* prog) {
    std::printf(
        "usage: %s <image> <number of colors> [options]\n"
        "  --block B       sample one pixel per BxB block (default 4)\n"
        "  --seed N        random seed (default 1)\n"
        "  --max-iter N    maximum E-M iterations (default 10)\n"
        "  --out DIR       write <name>_k<K>.png and <name>_k<K>_palette.png into DIR\n"
        "  --headless      never open windows (automatic when no display is available)\n",
        prog);
}

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help") || args.positional().size() < 2) {
            usage(argv[0]);
            return args.has("help") ? 0 : 1;
        }
        const std::string path = args.positional()[0];
        const int ncolors = std::stoi(args.positional()[1]);
        if (ncolors < 1 || ncolors > 66000) {
            std::fprintf(stderr, "error: number of colors must be in [1, 66000]\n");
            return 2;
        }
        const int B = args.getInt("block", 4);
        const unsigned seed = static_cast<unsigned>(args.getInt("seed", 1));
        const int maxIter = args.getInt("max-iter", 10);

        cv::Mat src = cv::imread(path);
        if (src.empty()) {
            std::fprintf(stderr, "error: unable to read image '%s'\n", path.c_str());
            return 2;
        }

        std::mt19937 rng(seed);
        std::vector<cv::Vec3b> data = sampleColors(src, B, rng);
        if (data.size() < static_cast<size_t>(ncolors)) {
            std::fprintf(stderr, "error: only %zu samples for %d colors; lower --block or K\n",
                         data.size(), ncolors);
            return 2;
        }
        std::vector<cv::Vec3b> means;
        std::vector<int> labels(data.size());
        if (kmeans(data, means, labels.data(), ncolors, maxIter, 0, seed)) return 2;

        cv::Mat dst = quantize(src, means);
        const std::vector<double> shares = paletteShares(dst, means);
        const double mse = meanSquaredError(src, dst);

        std::printf("%s: %dx%d, K=%d, %zu samples, MSE %.1f\n", path.c_str(), src.cols, src.rows,
                    ncolors, data.size(), mse);
        for (size_t k = 0; k < means.size(); ++k) {
            std::printf("  %s  %5.1f%%\n", toHex(means[k]).c_str(), 100.0 * shares[k]);
        }

        cvc::Viewer viewer = cvc::Viewer::fromArgs(args);
        if (args.has("out")) {
            const std::string stem = std::filesystem::path(path).stem().string() + "_k" + std::to_string(ncolors);
            cvc::writeImage(args.get("out") + "/" + stem + ".png", dst);
            cvc::writeImage(args.get("out") + "/" + stem + "_palette.png", paletteStrip(means, shares));
        }
        if (!viewer.headless()) {
            cv::namedWindow("Original", cv::WINDOW_NORMAL);
            cv::resizeWindow("Original", 800, 600);
            cv::imshow("Original", src);
            cv::namedWindow("clustered", cv::WINDOW_NORMAL);
            cv::resizeWindow("clustered", 800, 600);
            cv::imshow("clustered", dst);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
