// Small helpers shared by the command line tools: argument parsing, repo
// relative default paths, frame sources (camera / video / image) and a viewer
// that falls back to writing PNG files when no display is available.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cvc {

// ---- paths ---------------------------------------------------------------

// Root of the repository: $CVCODES_ROOT if set, else the source directory this
// binary was built from, else ".".
std::string repoRoot();
// repoRoot() + "/" + relative (no existence check).
std::string repoPath(const std::string& relative);

// Image files (jpg, jpeg, png, ppm, pgm, bmp, tif, tiff; case-insensitive) in
// `dir`, sorted by name. Returns an empty vector if the directory is missing.
std::vector<std::string> listImages(const std::string& dir);

// imwrite that creates missing parent directories. Returns false on failure.
bool writeImage(const std::string& path, const cv::Mat& image);

// ---- command line --------------------------------------------------------

// Minimal "--key value" / "--key=value" / "--flag" / positional parser.
// Anything not starting with "--" is positional (so "-1" is a value).
class Args {
public:
    // `flags`: option names (without dashes) that never take a value.
    Args(int argc, const char* const* argv, std::set<std::string> flags = {});

    bool has(const std::string& name) const;
    std::string get(const std::string& name, const std::string& fallback = "") const;
    // Throw std::invalid_argument when the value is present but malformed.
    int getInt(const std::string& name, int fallback) const;
    double getDouble(const std::string& name, double fallback) const;

    const std::vector<std::string>& positional() const { return positional_; }
    const std::string& program() const { return program_; }

private:
    std::string program_;
    std::map<std::string, std::string> options_;
    std::vector<std::string> positional_;
};

// ---- frames in, frames out ----------------------------------------------

// Open a camera ("0", "1", ...), a video file, or an image / image-sequence
// pattern ("frames/%04d.png"). Throws std::runtime_error when it cannot be opened.
cv::VideoCapture openSource(const std::string& spec);

// True when a GUI can be shown (Linux: DISPLAY or WAYLAND_DISPLAY is set).
bool displayAvailable();

// Shows frames in windows, or - in headless mode - writes them to
// "<outDir>/<window>_<index>.png".
class Viewer {
public:
    Viewer(bool headless, std::string outDir = "", int saveEvery = 1);
    // Reads --headless, --out DIR and --save-every N. Headless is also chosen
    // automatically when no display is available.
    static Viewer fromArgs(const Args& args, const std::string& defaultOutDir = "");

    bool headless() const { return headless_; }
    const std::string& outDir() const { return outDir_; }
    int savedCount() const { return saved_; }

    void show(const std::string& window, const cv::Mat& image);
    // Returns the pressed key, or -1 (never blocks) in headless mode.
    int waitKey(int delayMs) const;

private:
    bool headless_;
    std::string outDir_;
    int saveEvery_;
    int saved_ = 0;
    std::map<std::string, int> counters_;
};

}  // namespace cvc
