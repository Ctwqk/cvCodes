#include "dbtools.h"

#include <stdexcept>

#include "app_io.h"
#include "dnnEmbedding.h"

FeatureKind parseFeatureKind(const std::string& name) {
    if (name == "shape") return FeatureKind::Shape;
    if (name == "dnn") return FeatureKind::Dnn;
    throw std::invalid_argument("--features must be 'shape' or 'dnn', got '" + name + "'");
}

DistanceKind parseDistanceKind(const std::string& name) {
    if (name == "cosine") return DistanceKind::Cosine;
    if (name == "euclid") return DistanceKind::Euclidean;
    if (name == "scaled") return DistanceKind::ScaledEuclidean;
    throw std::invalid_argument("--metric must be cosine, euclid or scaled, got '" + name + "'");
}

std::string defaultDbPath(FeatureKind kind) {
    return cvc::repoPath(kind == FeatureKind::Shape ? "project3/trainset/shape_features.csv"
                                                    : "project3/trainset/dnn_features.csv");
}

std::string defaultModelPath() { return cvc::repoPath("project3/NetWorkThings/or2d-normmodel-007.onnx"); }

std::vector<float> regionFeature(const RegionInfo& region, FeatureKind kind, cv::dnn::Net* net) {
    if (kind == FeatureKind::Shape) return region.features;
    if (!net) throw std::invalid_argument("the DNN embedding needs a loaded network");
    return regionEmbedding(region.mask, region.box, *net);
}
