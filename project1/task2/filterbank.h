// Table of all project-1 effects, shared by the image and the video viewer so
// that both use the same hot keys and can also be driven by name from the
// command line (headless batch mode).
#ifndef FILTERBANK_H
#define FILTERBANK_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct FilterEntry {
    char key;          // interactive hot key
    const char* name;  // name used on the command line
    const char* help;  // one-line description
    // Applies the effect to a BGR image; `out` is always a displayable 8-bit
    // image. Returns 0 on success, non-zero when it cannot run (for example
    // the face cascade is missing).
    int (*apply)(const cv::Mat& in, cv::Mat& out);
};

const std::vector<FilterEntry>& filterBank();
const FilterEntry* findFilterByKey(int key);
const FilterEntry* findFilterByName(const std::string& name);

#endif
