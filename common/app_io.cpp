#include "app_io.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

#ifndef CVC_REPO_ROOT
#define CVC_REPO_ROOT "."
#endif

namespace cvc {

// ---- paths ---------------------------------------------------------------

std::string repoRoot() {
    if (const char* env = std::getenv("CVCODES_ROOT")) {
        if (*env) return env;
    }
    return CVC_REPO_ROOT;
}

std::string repoPath(const std::string& relative) {
    return (fs::path(repoRoot()) / relative).string();
}

std::vector<std::string> listImages(const std::string& dir) {
    static const std::set<std::string> exts = {".jpg", ".jpeg", ".png", ".ppm", ".pgm",
                                               ".bmp", ".tif",  ".tiff"};
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (exts.count(ext)) out.push_back(it->path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool writeImage(const std::string& path, const cv::Mat& image) {
    const fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }
    try {
        return cv::imwrite(path, image);
    } catch (const cv::Exception&) {
        return false;
    }
}

// ---- command line --------------------------------------------------------

Args::Args(int argc, const char* const* argv, std::set<std::string> flags) {
    if (argc > 0 && argv[0]) program_ = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string tok = argv[i];
        if (tok.rfind("--", 0) != 0 || tok.size() == 2) {
            positional_.push_back(tok);
            continue;
        }
        std::string name = tok.substr(2);
        const size_t eq = name.find('=');
        if (eq != std::string::npos) {
            options_[name.substr(0, eq)] = name.substr(eq + 1);
        } else if (flags.count(name) || i + 1 >= argc ||
                   std::string(argv[i + 1]).rfind("--", 0) == 0) {
            options_[name] = "";
        } else {
            options_[name] = argv[++i];
        }
    }
}

bool Args::has(const std::string& name) const { return options_.count(name) > 0; }

std::string Args::get(const std::string& name, const std::string& fallback) const {
    auto it = options_.find(name);
    return (it == options_.end() || it->second.empty()) ? fallback : it->second;
}

int Args::getInt(const std::string& name, int fallback) const {
    const std::string v = get(name);
    if (v.empty()) return fallback;
    size_t used = 0;
    try {
        const int x = std::stoi(v, &used);
        if (used == v.size()) return x;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("--" + name + ": expected an integer, got '" + v + "'");
}

double Args::getDouble(const std::string& name, double fallback) const {
    const std::string v = get(name);
    if (v.empty()) return fallback;
    size_t used = 0;
    try {
        const double x = std::stod(v, &used);
        if (used == v.size()) return x;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("--" + name + ": expected a number, got '" + v + "'");
}

// ---- frames in, frames out ----------------------------------------------

cv::VideoCapture openSource(const std::string& spec) {
    cv::VideoCapture cap;
    const bool numeric = !spec.empty() && std::all_of(spec.begin(), spec.end(), [](unsigned char c) {
                             return std::isdigit(c) != 0;
                         });
    if (numeric) {
        cap.open(std::stoi(spec));
    } else {
        cap.open(spec);
    }
    if (!cap.isOpened()) {
        throw std::runtime_error("cannot open video source '" + spec + "'");
    }
    return cap;
}

bool displayAvailable() {
#ifdef __linux__
    const char* d = std::getenv("DISPLAY");
    const char* w = std::getenv("WAYLAND_DISPLAY");
    return (d && *d) || (w && *w);
#else
    return true;
#endif
}

Viewer::Viewer(bool headless, std::string outDir, int saveEvery)
    : headless_(headless), outDir_(std::move(outDir)), saveEvery_(std::max(1, saveEvery)) {}

Viewer Viewer::fromArgs(const Args& args, const std::string& defaultOutDir) {
    const bool headless = args.has("headless") || !displayAvailable();
    return Viewer(headless, args.get("out", headless ? defaultOutDir : ""),
                  args.getInt("save-every", 1));
}

void Viewer::show(const std::string& window, const cv::Mat& image) {
    if (image.empty()) return;
    if (!headless_) {
        cv::imshow(window, image);
        return;
    }
    if (outDir_.empty()) return;
    const int idx = counters_[window]++;
    if (idx % saveEvery_ != 0) return;
    cv::Mat out = image;
    if (image.depth() != CV_8U) {
        cv::normalize(image, out, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    char name[64];
    std::snprintf(name, sizeof name, "_%05d.png", idx);
    if (writeImage((fs::path(outDir_) / (window + name)).string(), out)) ++saved_;
}

int Viewer::waitKey(int delayMs) const {
    return headless_ ? -1 : cv::waitKey(delayMs);
}

}  // namespace cvc
