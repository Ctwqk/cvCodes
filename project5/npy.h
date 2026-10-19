// Minimal NumPy .npy reader / writer (replaces the external cnpy dependency, which the
// original CMake file pulled from a hard-coded path on the author's machine).
//
// Reads little-endian float32/float64/int32/int64/uint8 C-ordered arrays of any rank
// (converted to float); writes float32.
#ifndef NPY_H
#define NPY_H

#include <string>
#include <vector>

namespace p5 {

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<float> data;  // row-major
    size_t size() const {
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }
};

bool readNpy(const std::string& path, NpyArray& out, std::string* err = nullptr);
// Writes format version 1.0, dtype '<f4'. `data.size()` must equal the product of `shape`.
bool writeNpy(const std::string& path, const std::vector<size_t>& shape, const std::vector<float>& data,
              std::string* err = nullptr);

}  // namespace p5

#endif
