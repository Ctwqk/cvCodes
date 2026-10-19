// Feature CSV files: one row per image, "<name>,<f0>,<f1>,...".
//
// Replaces the course-supplied csv_util (fixed 256-byte buffers, leaked
// filename arrays, int/char EOF confusion) with a std::string based API.
#pragma once
#include <string>
#include <vector>

namespace cvc {

struct FeatureTable {
    std::vector<std::string> names;          // first column of every row
    std::vector<std::vector<float>> data;    // remaining columns of every row
    size_t size() const { return names.size(); }
};

// Append one row to `path` (or create/truncate the file when `truncate` is
// true). Returns false (and fills `err` if given) on failure. `name` must not
// contain ',' or a newline.
bool appendFeatureRow(const std::string& path, const std::string& name,
                      const std::vector<float>& values, bool truncate = false,
                      std::string* err = nullptr);

// Read a whole file. Blank lines are skipped, CRLF is accepted. Every row must
// hold only numeric values after the name; otherwise false is returned and
// `err` describes the offending line. Rows may have different lengths.
bool readFeatureTable(const std::string& path, FeatureTable& table, std::string* err = nullptr);

// Write the full table (truncating `path`).
bool writeFeatureTable(const std::string& path, const FeatureTable& table, std::string* err = nullptr);

// Parse a single CSV line into name + values (exposed for tests).
bool parseFeatureLine(const std::string& line, std::string& name, std::vector<float>& values,
                      std::string* err = nullptr);

// Format a single row without trailing newline (exposed for tests).
std::string formatFeatureLine(const std::string& name, const std::vector<float>& values);

// Last path component ("a/b/pic.0001.jpg" -> "pic.0001.jpg").
std::string baseName(const std::string& path);

}  // namespace cvc
