// Helpers shared by buildDb / recognize / evalDb / vidDis: feature kinds, default paths, embedding model.
#ifndef DBTOOLS_H
#define DBTOOLS_H

#include <opencv2/dnn.hpp>
#include <string>

#include "pipeline.h"
#include "recognition.h"

enum class FeatureKind { Shape, Dnn };

// "shape" | "dnn" (throws std::invalid_argument otherwise)
FeatureKind parseFeatureKind(const std::string& name);
// "cosine" | "euclid" | "scaled"
DistanceKind parseDistanceKind(const std::string& name);

std::string defaultDbPath(FeatureKind kind);   // project3/trainset/shape_features.csv | dnn_features.csv
std::string defaultModelPath();                // project3/NetWorkThings/or2d-normmodel-007.onnx

// Feature vector of one region, either the 3 shape features or the DNN embedding.
// `net` may be null for FeatureKind::Shape.
std::vector<float> regionFeature(const RegionInfo& region, FeatureKind kind, cv::dnn::Net* net);

#endif
