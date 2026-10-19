/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task: compares the pixel-wise
  5x5 blur (blur5x5_1) with the separable version (blur5x5_2).

  usage: timeBlur <image> [--repeat N] [--max-width W]
*/
#include <chrono>
#include <cstdio>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "filter.h"

namespace {
// seconds per call
template <typename F>
double timeIt(int repeat, F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / repeat;
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        if (args.has("help") || args.positional().empty()) {
            std::printf("Usage %s <image filename> [--repeat N] [--max-width W]\n", argv[0]);
            return args.has("help") ? 0 : 1;
        }
        cv::Mat src = cv::imread(args.positional()[0]);
        if (src.empty()) {
            std::fprintf(stderr, "Unable to read image %s\n", args.positional()[0].c_str());
            return 2;
        }
        const int maxWidth = args.getInt("max-width", 0);
        if (maxWidth > 0 && src.cols > maxWidth) {
            cv::resize(src, src, cv::Size(maxWidth, src.rows * maxWidth / src.cols), 0, 0, cv::INTER_AREA);
        }
        const int repeat = args.getInt("repeat", 10);
        if (repeat < 1) throw std::invalid_argument("--repeat must be positive");
        cv::Mat dst1, dst2;
        const double t1 = timeIt(repeat, [&] { blur5x5_1(src, dst1); });
        const double t2 = timeIt(repeat, [&] { blur5x5_2(src, dst2); });
        std::printf("image %dx%d, %d repeats\n", src.cols, src.rows, repeat);
        std::printf("Time per image (1): %.4f seconds\n", t1);
        std::printf("Time per image (2): %.4f seconds\n", t2);
        std::printf("speed-up of the separable version: %.1fx, outputs identical: %s\n", t1 / t2,
                    cv::norm(dst1, dst2, cv::NORM_INF) == 0 ? "yes" : "no");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
