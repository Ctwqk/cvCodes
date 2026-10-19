#include "npy.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace p5 {

namespace {

bool fail(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}

// value of 'key' in the header dictionary, e.g. 'descr': '<f4'
std::string headerValue(const std::string& header, const std::string& key) {
    const size_t k = header.find("'" + key + "'");
    if (k == std::string::npos) return "";
    size_t pos = header.find(':', k);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < header.size() && header[pos] == ' ') pos++;
    if (pos >= header.size()) return "";
    if (header[pos] == '(') {
        const size_t end = header.find(')', pos);
        return end == std::string::npos ? "" : header.substr(pos, end - pos + 1);
    }
    if (header[pos] == '\'') {
        const size_t end = header.find('\'', pos + 1);
        return end == std::string::npos ? "" : header.substr(pos + 1, end - pos - 1);
    }
    const size_t end = header.find_first_of(",}", pos);
    std::string v = header.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    while (!v.empty() && v.back() == ' ') v.pop_back();
    return v;
}

template <typename T>
void convert(const char* raw, size_t n, std::vector<float>& out) {
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        T v;
        std::memcpy(&v, raw + i * sizeof(T), sizeof(T));
        out[i] = static_cast<float>(v);
    }
}

}  // namespace

bool readNpy(const std::string& path, NpyArray& out, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail(err, "cannot open '" + path + "'");
    char magic[6];
    in.read(magic, 6);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) return fail(err, path + ": not a .npy file");
    unsigned char ver[2]{};
    in.read(reinterpret_cast<char*>(ver), 2);
    uint32_t headerLen = 0;
    if (ver[0] == 1) {
        unsigned char b[2]{};
        in.read(reinterpret_cast<char*>(b), 2);
        headerLen = b[0] | (b[1] << 8);
    } else if (ver[0] == 2 || ver[0] == 3) {
        unsigned char b[4]{};
        in.read(reinterpret_cast<char*>(b), 4);
        headerLen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
    } else {
        return fail(err, path + ": unsupported .npy version");
    }
    if (!in || headerLen == 0 || headerLen > (1u << 20)) return fail(err, path + ": bad header length");
    std::string header(headerLen, '\0');
    in.read(&header[0], headerLen);
    if (!in) return fail(err, path + ": truncated header");

    const std::string descr = headerValue(header, "descr");
    if (headerValue(header, "fortran_order") == "True") return fail(err, path + ": Fortran-ordered arrays are not supported");
    const std::string shapeText = headerValue(header, "shape");
    if (shapeText.empty()) return fail(err, path + ": no shape in header");
    if (shapeText.find_first_not_of("() ,0123456789") != std::string::npos)
        return fail(err, path + ": invalid shape");
    out.shape.clear();
    {
        std::string digits;
        for (char c : shapeText) {
            if (c >= '0' && c <= '9') {
                digits += c;
            } else if (!digits.empty()) {
                try { out.shape.push_back(std::stoull(digits)); }
                catch (const std::exception&) { return fail(err, path + ": shape dimension out of range"); }
                digits.clear();
            }
        }
    }
    size_t n = 1; // empty shape () -> 1 element
    for (size_t dim : out.shape) {
        if (dim > (1u << 26) || (dim && n > (1u << 26) / dim)) return fail(err, path + ": array too large");
        n *= dim;
    }

    size_t itemSize = 0;
    if (descr == "<f4" || descr == "<i4" || descr == "<u4") itemSize = 4;
    else if (descr == "<f8" || descr == "<i8" || descr == "<u8") itemSize = 8;
    else if (descr == "|u1" || descr == "|i1" || descr == "<u1" || descr == "|b1") itemSize = 1;
    else return fail(err, path + ": unsupported dtype '" + descr + "'");

    std::vector<char> raw(n * itemSize);
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (static_cast<size_t>(in.gcount()) != raw.size()) return fail(err, path + ": truncated data");

    if (descr == "<f4") convert<float>(raw.data(), n, out.data);
    else if (descr == "<f8") convert<double>(raw.data(), n, out.data);
    else if (descr == "<i4") convert<int32_t>(raw.data(), n, out.data);
    else if (descr == "<u4") convert<uint32_t>(raw.data(), n, out.data);
    else if (descr == "<i8") convert<int64_t>(raw.data(), n, out.data);
    else if (descr == "<u8") convert<uint64_t>(raw.data(), n, out.data);
    else if (descr == "|i1") convert<int8_t>(raw.data(), n, out.data);
    else convert<uint8_t>(raw.data(), n, out.data);
    return true;
}

bool writeNpy(const std::string& path, const std::vector<size_t>& shape, const std::vector<float>& data,
              std::string* err) {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    if (n != data.size()) return fail(err, "writeNpy: shape does not match the data size");
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++) dict << (i ? ", " : "") << shape[i];
    if (shape.size() == 1) dict << ",";
    dict << "), }";
    std::string header = dict.str();
    // magic(6) + version(2) + length(2) + header + '\n' must be a multiple of 64
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return fail(err, "cannot open '" + path + "' for writing");
    out.write("\x93NUMPY\x01\x00", 8);
    const uint16_t len = static_cast<uint16_t>(header.size());
    const unsigned char lenBytes[2] = {static_cast<unsigned char>(len & 0xff), static_cast<unsigned char>(len >> 8)};
    out.write(reinterpret_cast<const char*>(lenBytes), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(out) || fail(err, "write failed");
}

}  // namespace p5
