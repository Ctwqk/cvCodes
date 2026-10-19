/* Taiwei Cui, April 2024: build an image-pattern descriptor database. */
#include "saveOrReadPattern.hpp"
#include "app_io.h"

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        if (args.has("help")) {
            std::cout << "usage: buildPatterns [--images DIR] [--out FILE]\n";
            return 0;
        }
        const auto out = args.get("out", "results/patterns.txt");
        const auto parent = std::filesystem::path(out).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        return establishPatterns(args.get("images", cvc::repoPath("project1")), out, 400, 1) == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
