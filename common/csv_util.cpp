#include "csv_util.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace cvc {
namespace {
bool prepareParent(const std::string& path, std::string* err) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (!ec) return true;
    if (err) *err = "cannot create directory '" + parent.string() + "': " + ec.message();
    return false;
}
}  // namespace


std::string baseName(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string formatFeatureLine(const std::string& name, const std::vector<float>& values) {
    std::string line = name;
    char buf[32];
    for (float v : values) {
        std::snprintf(buf, sizeof buf, ",%.7g", static_cast<double>(v));
        line += buf;
    }
    return line;
}

bool parseFeatureLine(const std::string& rawLine, std::string& name, std::vector<float>& values,
                      std::string* err) {
    std::string line = rawLine;
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    values.clear();
    const size_t first = line.find(',');
    name = line.substr(0, first);
    if (name.empty()) {
        if (err) *err = "empty name column";
        return false;
    }
    size_t pos = first;
    while (pos != std::string::npos) {
        const size_t start = pos + 1;
        const size_t next = line.find(',', start);
        const std::string tok = line.substr(start, next == std::string::npos ? next : next - start);
        char* end = nullptr;
        errno = 0;
        const float v = std::strtof(tok.c_str(), &end);
        if (tok.empty() || end == tok.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
            if (err) *err = "bad numeric field '" + tok + "'";
            return false;
        }
        values.push_back(v);
        pos = next;
    }
    return true;
}

bool appendFeatureRow(const std::string& path, const std::string& name,
                      const std::vector<float>& values, bool truncate, std::string* err) {
    if (name.find_first_of(",\r\n") != std::string::npos) {
        if (err) *err = "name must not contain ',' or newlines: " + name;
        return false;
    }
    if (!prepareParent(path, err)) return false;
    std::ofstream out(path, truncate ? std::ios::trunc : std::ios::app);
    if (!out) {
        if (err) *err = "cannot open '" + path + "' for writing";
        return false;
    }
    out << formatFeatureLine(name, values) << '\n';
    return static_cast<bool>(out);
}

bool writeFeatureTable(const std::string& path, const FeatureTable& table, std::string* err) {
    if (table.names.size() != table.data.size()) {
        if (err) *err = "names/data size mismatch";
        return false;
    }
    if (!prepareParent(path, err)) return false;
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        if (err) *err = "cannot open '" + path + "' for writing";
        return false;
    }
    for (size_t i = 0; i < table.size(); ++i) {
        if (table.names[i].find_first_of(",\r\n") != std::string::npos) {
            if (err) *err = "name must not contain ',' or newlines: " + table.names[i];
            return false;
        }
        out << formatFeatureLine(table.names[i], table.data[i]) << '\n';
    }
    return static_cast<bool>(out);
}

bool readFeatureTable(const std::string& path, FeatureTable& table, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "cannot open '" + path + "'";
        return false;
    }
    table.names.clear();
    table.data.clear();
    std::string line;
    size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty() || line == "\r") continue;
        std::string name;
        std::vector<float> values;
        std::string lineErr;
        if (!parseFeatureLine(line, name, values, &lineErr)) {
            if (err) *err = path + ":" + std::to_string(lineNo) + ": " + lineErr;
            return false;
        }
        table.names.push_back(std::move(name));
        table.data.push_back(std::move(values));
    }
    return true;
}

}  // namespace cvc
