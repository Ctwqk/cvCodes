#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include <cstdlib>
#include <filesystem>
#include <random>
#include <stdexcept>

// Give each executable a private scratch directory; individual fixtures may
// safely create/remove their own children without touching another test run.
int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const auto parent = fs::temp_directory_path();
    fs::path scratch;
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
        scratch = parent / ("pu-cvcodes-tests-" + std::to_string(random()));
        if (fs::create_directory(scratch)) break;
        scratch.clear();
    }
    if (scratch.empty()) throw std::runtime_error("cannot create test scratch directory");
#ifdef _WIN32
    _putenv_s("TMP", scratch.string().c_str());
    _putenv_s("TEMP", scratch.string().c_str());
#else
    setenv("TMPDIR", scratch.string().c_str(), 1);
#endif
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    fs::remove_all(scratch);
    return result;
}
