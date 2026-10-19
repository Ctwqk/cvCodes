#include "idx.h"

#include <fstream>

namespace p5 {

namespace {
bool fail(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}
bool readBE32(std::istream& in, uint32_t& v) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) return false;
    v = (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) | (static_cast<uint32_t>(b[2]) << 8) | b[3];
    return true;
}
}  // namespace

bool readIdxImages(const std::string& path, std::vector<cv::Mat>& images, std::string* err, size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail(err, "cannot open '" + path + "'");
    uint32_t magic = 0, count = 0, rows = 0, cols = 0;
    if (!readBE32(in, magic) || !readBE32(in, count) || !readBE32(in, rows) || !readBE32(in, cols)) {
        return fail(err, path + ": truncated header");
    }
    if (magic != 0x00000803) return fail(err, path + ": not an idx3-ubyte image file (magic " + std::to_string(magic) + ")");
    if (rows == 0 || cols == 0 || rows > 4096 || cols > 4096) return fail(err, path + ": implausible image size");
    const size_t n = (limit > 0 && limit < count) ? limit : count;
    images.clear();
    if (n > (1u << 20)) return fail(err, path + ": too many images");
    images.reserve(n);
    for (size_t i = 0; i < n; i++) {
        cv::Mat m(static_cast<int>(rows), static_cast<int>(cols), CV_8UC1);
        in.read(reinterpret_cast<char*>(m.data), static_cast<std::streamsize>(m.total()));
        if (static_cast<size_t>(in.gcount()) != m.total()) return fail(err, path + ": file ends after " + std::to_string(i) + " images");
        images.push_back(m);
    }
    return true;
}

bool readIdxLabels(const std::string& path, std::vector<uint8_t>& labels, std::string* err, size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail(err, "cannot open '" + path + "'");
    uint32_t magic = 0, count = 0;
    if (!readBE32(in, magic) || !readBE32(in, count)) return fail(err, path + ": truncated header");
    if (magic != 0x00000801) return fail(err, path + ": not an idx1-ubyte label file (magic " + std::to_string(magic) + ")");
    const size_t n = (limit > 0 && limit < count) ? limit : count;
    if (n > (1u << 20)) return fail(err, path + ": too many labels");
    labels.resize(n);
    in.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(n));
    if (static_cast<size_t>(in.gcount()) != n) return fail(err, path + ": truncated labels");
    return true;
}

}  // namespace p5
