// Content-based image retrieval on top of features.h: feature registry,
// database building, ranking and result montages.
#ifndef RETRIEVAL_H
#define RETRIEVAL_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "csv_util.h"

enum class Metric {
    SSD,                  // sum of squared differences (smaller = more similar)
    HistIntersection,     // 1 - normalised histogram intersection
    Cosine,               // 1 - cosine similarity
};

struct FeatureSpec {
    std::string name;         // command line name
    std::string dbFile;       // CSV file name inside the database directory
    Metric metric;
    bool needsDnn;            // uses the precomputed ResNet18 embedding of the image
    std::string description;
};

// 7x7central, histogram, multihist, texturecolor, gabor, dnn, mycustom
const std::vector<FeatureSpec>& featureSpecs();
const FeatureSpec* findFeatureSpec(const std::string& name);

// File holding the precomputed ResNet18 embeddings (provided with the course data).
extern const char* const kDnnCsvName;

float distanceFor(Metric metric, const std::vector<float>& a, const std::vector<float>& b);

// Feature vector of `bgr` for `spec`. `imageName` (a path or a base name) is
// used to look up the embedding in `dnn` for the features that need it.
// Returns false and fills `err` on failure.
bool computeFeature(const FeatureSpec& spec, const cv::Mat& bgr, const std::string& imageName,
                    const cvc::FeatureTable* dnn, std::vector<float>& out, std::string& err);

// Compute the feature of every image in `imageDir` (sorted by name, optional
// `limit`) and write the database CSV (rows are named by base name).
// Returns the number of images written, or -1 on error.
int buildDatabase(const FeatureSpec& spec, const std::string& imageDir, const std::string& csvPath,
                  const cvc::FeatureTable* dnn, int limit, std::string& err);

struct Match {
    size_t index;       // row in the database
    std::string name;
    float distance;     // smaller = more similar
};

// The k closest rows (or the k farthest when `farthest`), ties broken by row
// order. Rows named `excludeName` (compared by base name) are skipped, so a
// query that is part of the database does not match itself. Rows whose size
// differs from the query are skipped as well.
std::vector<Match> rankMatches(const std::vector<float>& query, const cvc::FeatureTable& db,
                               Metric metric, size_t k, const std::string& excludeName = "",
                               bool farthest = false);

// Query tile followed by one tile per match, captions "<rank> <name> d=<distance>".
cv::Mat makeMontage(const cv::Mat& query, const std::string& queryName,
                    const std::vector<cv::Mat>& images, const std::vector<Match>& matches,
                    int tileWidth = 240, int tileHeight = 192);

#endif
