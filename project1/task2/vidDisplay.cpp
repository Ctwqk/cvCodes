/*
Author: Taiwei Cui
Time: 2024-1-22

Live video with the project-1 filters.

  vidDisplay [--source 0|video.mp4|image.jpg] [--filter NAME] [--out DIR] [--max-frames N]

Interactive keys (see `imgDisplay --list` for the filter keys):
  o g h b n x y m l f c r k e p   choose a filter (o = original)
  + / -   brightness +-5 (range -50..50)      > / <   contrast +-0.05 (range 0.5..2)
  a       reset brightness and contrast       s       save the current frame
  q       quit

Headless: give --filter and --out; every processed frame is written to
DIR/Video_<n>.png (use --save-every N / --max-frames N to limit the output).
*/
#include <cstdio>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "filter.h"
#include "filterbank.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help")) {
            std::printf(
                "usage: %s [--source 0|video|image] [--filter NAME] [--out DIR] [--max-frames N] "
                "[--save-every N] [--headless]\n",
                argv[0]);
            return 0;
        }
        cv::VideoCapture cap = cvc::openSource(args.get("source", "0"));
        cv::Size size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        std::printf("Expected size: %d %d\n", size.width, size.height);

        cvc::Viewer viewer = cvc::Viewer::fromArgs(args, "results/vidDisplay");
        const FilterEntry* current = findFilterByName(args.get("filter", "original"));
        if (!current) {
            std::fprintf(stderr, "error: unknown filter '%s'\n", args.get("filter").c_str());
            return 1;
        }
        const int maxFrames = args.getInt("max-frames", -1);
        int brightScaler = 0;
        float contrastScaler = 1;

        cv::Mat frame, shown;
        for (int n = 0; maxFrames < 0 || n < maxFrames; n++) {
            cap >> frame;  // a new frame from the camera / file, treated as a stream
            if (frame.empty()) {
                std::printf("frame is empty\n");
                break;
            }
            if (current->apply(frame, shown)) shown = frame;  // e.g. no face cascade: show the original
            brightAndContrast(shown, shown, contrastScaler, brightScaler);
            viewer.show("Video", shown);

            const int key = viewer.waitKey(10);
            if (key < 0) continue;
            if (key == 'q') break;
            if (const FilterEntry* e = findFilterByKey(key)) {
                current = e;
            } else if (key == 's') {
                int idx = 0;
                std::string filename = "savedImg(0).jpg";
                while (std::ifstream(filename).good()) filename = "savedImg(" + std::to_string(++idx) + ").jpg";
                if (cv::imwrite(filename, shown)) std::printf("saved %s\n", filename.c_str());
            } else if (key == '+' && brightScaler < 50) {
                brightScaler += 5;
            } else if (key == '-' && brightScaler > -50) {
                brightScaler -= 5;
            } else if (key == '>' && contrastScaler < 2) {
                contrastScaler += 0.05f;
            } else if (key == '<' && contrastScaler > 0.5) {
                contrastScaler -= 0.05f;
            } else if (key == 'a') {
                brightScaler = 0;
                contrastScaler = 1;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
