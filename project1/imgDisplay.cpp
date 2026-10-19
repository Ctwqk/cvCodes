/*
  Display an image and apply the project-1 filters to it.

  Interactive (needs a display):  imgDisplay <image>
    keys: o g h p b n x y m l f c r k e (see --list), +/- brightness, </> contrast,
          a reset brightness/contrast, s save current view, q quit

  Headless batch mode:  imgDisplay <image> --all --out DIR
                        imgDisplay <image> --filter sepia --filter sobelx --out DIR
*/
#include <chrono>
#include <cstdio>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "filter.h"
#include "filterbank.h"

namespace {

void usage(const char* prog) {
    std::printf(
        "usage: %s <image> [options]\n"
        "  --all              apply every filter and write one PNG per filter (needs --out)\n"
        "  --filter NAME      apply only this filter (repeatable via comma list: sepia,sobelx)\n"
        "  --out DIR          output directory for --all / --filter / 's' key\n"
        "  --max-width W      shrink the input to at most W pixels wide first\n"
        "  --brightness B     add B to every channel after filtering (default 0)\n"
        "  --contrast C       multiply by C after filtering (default 1)\n"
        "  --list             list the available filters and exit\n"
        "  --headless         never open windows (automatic when no display is available)\n",
        prog);
}

void listFilters() {
    for (const FilterEntry& e : filterBank()) std::printf("  %c  %-13s %s\n", e.key, e.name, e.help);
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t pos = s.find(sep, start);
        parts.push_back(s.substr(start, pos == std::string::npos ? pos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return parts;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"all", "headless", "list", "help"});
        if (args.has("list")) {
            listFilters();
            return 0;
        }
        if (args.has("help") || args.positional().empty()) {
            usage(argv[0]);
            return args.has("help") ? 0 : 1;
        }
        const std::string path = args.positional()[0];
        cv::Mat picture = cv::imread(path, cv::IMREAD_COLOR);
        if (picture.empty()) {
            std::fprintf(stderr, "error: unable to read image '%s'\n", path.c_str());
            return 2;
        }
        const int maxWidth = args.getInt("max-width", 0);
        if (maxWidth > 0 && picture.cols > maxWidth) {
            cv::resize(picture, picture, cv::Size(maxWidth, picture.rows * maxWidth / picture.cols), 0, 0,
                       cv::INTER_AREA);
        }
        const int brightness = args.getInt("brightness", 0);
        const float contrast = static_cast<float>(args.getDouble("contrast", 1.0));
        const std::string outDir = args.get("out", "");

        auto render = [&](const FilterEntry& e, cv::Mat& shown) {
            cv::Mat filtered;
            if (e.apply(picture, filtered)) return -1;
            brightAndContrast(filtered, shown, contrast, brightness);
            return 0;
        };

        // ---- headless batch mode -------------------------------------------
        if (args.has("all") || args.has("filter")) {
            if (outDir.empty()) {
                std::fprintf(stderr, "error: --all/--filter need --out DIR\n");
                return 1;
            }
            std::vector<const FilterEntry*> todo;
            if (args.has("all")) {
                for (const FilterEntry& e : filterBank()) todo.push_back(&e);
            } else {
                for (const std::string& n : split(args.get("filter"), ',')) {
                    const FilterEntry* e = findFilterByName(n);
                    if (!e) {
                        std::fprintf(stderr, "error: unknown filter '%s' (try --list)\n", n.c_str());
                        return 1;
                    }
                    todo.push_back(e);
                }
            }
            int failures = 0;
            for (const FilterEntry* e : todo) {
                cv::Mat shown;
                const auto t0 = std::chrono::steady_clock::now();
                const int rc = render(*e, shown);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (rc) {
                    std::printf("%-13s skipped (not available)\n", e->name);
                    ++failures;
                    continue;
                }
                cvc::writeImage(outDir + "/" + e->name + ".png", shown);
                std::printf("%-13s %8.1f ms  -> %s/%s.png\n", e->name, ms, outDir.c_str(), e->name);
            }
            return 0;
        }

        // ---- interactive mode ----------------------------------------------
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args);
        if (viewer.headless()) {
            std::fprintf(stderr, "no display available: use --all or --filter NAME with --out DIR\n");
            return 1;
        }
        cv::namedWindow("picture", cv::WINDOW_NORMAL);
        cv::resizeWindow("picture", 800, 480);
        cv::Mat frame = picture;
        const FilterEntry* current = findFilterByName("original");
        for (;;) {
            const int c = cv::waitKey(10);
            if (c == 'q') break;
            if (const FilterEntry* e = findFilterByKey(c)) {
                current = e;
            } else if (c == 's') {
                std::string dir = outDir.empty() ? "." : outDir;
                int idx = 0;
                std::string filename;
                do {
                    filename = dir + "/savedImg(" + std::to_string(idx++) + ").jpg";
                } while (std::ifstream(filename).good());
                if (cvc::writeImage(filename, frame)) std::printf("saved %s\n", filename.c_str());
            }
            if (c >= 0) {
                cv::Mat shown;
                if (render(*current, shown) == 0) {
                    frame = shown;
                } else {
                    std::printf("filter '%s' is not available\n", current->name);
                    frame = picture;
                }
            }
            cv::imshow("picture", frame);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
