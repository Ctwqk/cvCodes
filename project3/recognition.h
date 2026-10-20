// Nearest-neighbour recognition of shape / embedding feature vectors (project 3, steps 5-6).
#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <string>
#include <vector>

#include "csv_util.h"

enum class DistanceKind {
    Cosine,           // 1 - cosine similarity (what the original vidDis used)
    Euclidean,        // plain Euclidean distance
    ScaledEuclidean,  // Euclidean after dividing every feature by its standard deviation in the database
};

struct ClassifierConfig {
    DistanceKind kind = DistanceKind::ScaledEuclidean;
    int k = 1;  // k-nearest-neighbour vote; ties go to the class of the closest neighbour
};

struct Classification {
    std::string label;          // "" when the database is empty / has no comparable rows
    float distance = 0.f;       // distance to the closest neighbour
    std::vector<size_t> neighbours;  // database rows used for the vote, closest first
};

// Row names are "label" or "label:source" (the source, usually an image name, is ignored for voting).
std::string labelOf(const std::string& rowName);

// Class label from a training image name: "ball2.jpg" -> "ball", "tri.jpg" -> "tri"
// (directory, extension and trailing digits are dropped).
std::string labelFromFilename(const std::string& path);

// Standard deviation of every column (population), optionally leaving out one row.
// The two-argument form uses the first row's length. With `dimensions`, only rows
// of that length participate, allowing a caller to cache scales for its queries.
std::vector<float> columnStdDev(const cvc::FeatureTable& db, long skipRow = -1);
std::vector<float> columnStdDev(const cvc::FeatureTable& db, long skipRow, size_t dimensions);

// Classify `feature`; `scale` (see columnStdDev) is only used for ScaledEuclidean and is computed
// from comparable database rows (without `skipRow`) when empty.
// `skipRow` excludes one database row (leave-one-out), -1 for none.
Classification classify(const std::vector<float>& feature, const cvc::FeatureTable& db,
                        const ClassifierConfig& config, const std::vector<float>& scale = {}, long skipRow = -1);

struct ConfusionMatrix {
    std::vector<std::string> labels;       // sorted class names
    std::vector<std::vector<int>> counts;  // counts[truth][predicted]
    int total() const;
    int correct() const;
    double accuracy() const;
};

// Leave-one-out evaluation: every row is classified against all the others.
ConfusionMatrix leaveOneOut(const cvc::FeatureTable& db, const ClassifierConfig& config);

#endif
