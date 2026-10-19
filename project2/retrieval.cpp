#include "retrieval.h"

#include <algorithm>
#include <cstdio>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_io.h"
#include "distance.h"
#include "imgfeatures.h"

const char* const kDnnCsvName = "ResNet18_olym.csv";

const std::vector<FeatureSpec>& featureSpecs() {
    static const std::vector<FeatureSpec> specs = {
        {"7x7central", "7x7central.csv", Metric::SSD, false, "central 7x7 colour patch (147 values), SSD"},
        {"histogram", "histogram.csv", Metric::HistIntersection, false, "32x32 g/r chromaticity histogram, intersection"},
        {"multihist", "multihist.csv", Metric::HistIntersection, false, "five spatial chromaticity histograms, intersection"},
        {"texturecolor", "texturecolor.csv", Metric::HistIntersection, false, "Sobel orientation/magnitude histogram + colour histogram, intersection"},
        {"gabor", "gabor.csv", Metric::HistIntersection, false, "histogram of the Gabor-filtered greyscale image, intersection"},
        {"dnn", kDnnCsvName, Metric::SSD, true, "ResNet18 embedding (512 values, precomputed), SSD"},
        {"mycustom", "mycustom.csv", Metric::Cosine, true, "low-frequency DFT blocks + ResNet18 embedding, cosine"},
    };
    return specs;
}

const FeatureSpec* findFeatureSpec(const std::string& name) {
    for (const FeatureSpec& s : featureSpecs()) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

float distanceFor(Metric metric, const std::vector<float>& a, const std::vector<float>& b) {
    switch (metric) {
        case Metric::SSD: return cvc::sumSquaredDiff(a, b);
        case Metric::HistIntersection: return cvc::histogramIntersectionDistance(a, b);
        case Metric::Cosine: return cvc::cosineDistance(a, b);
    }
    return 0.f;
}

namespace {
const std::vector<float>* findEmbedding(const cvc::FeatureTable& table, const std::string& imageName) {
    const std::string base = cvc::baseName(imageName);
    for (size_t i = 0; i < table.size(); i++) {
        if (cvc::baseName(table.names[i]) == base) return &table.data[i];
    }
    return nullptr;
}
}  // namespace

bool computeFeature(const FeatureSpec& spec, const cv::Mat& bgr, const std::string& imageName,
                    const cvc::FeatureTable* dnn, std::vector<float>& out, std::string& err) {
    out.clear();
    int rc = 0;
    std::vector<float> extra;
    if (spec.needsDnn) {
        if (!dnn) {
            err = "feature '" + spec.name + "' needs the ResNet18 embedding table (" + kDnnCsvName + ")";
            return false;
        }
        const std::vector<float>* e = findEmbedding(*dnn, imageName);
        if (!e) {
            err = "no precomputed ResNet18 embedding for '" + cvc::baseName(imageName) +
                  "' (embeddings only exist for the images of the provided dataset)";
            return false;
        }
        extra = *e;
    }
    if (spec.name == "7x7central") {
        cv::Mat patch;
        rc = central7x7(bgr, patch);
        if (!rc) out = mat2vec(patch);
    } else if (spec.name == "histogram") {
        rc = histogram(bgr, out);
    } else if (spec.name == "multihist") {
        rc = multiHist(bgr, out);
    } else if (spec.name == "texturecolor") {
        std::vector<float> color;
        rc = texture(bgr, out) || histogram(bgr, color);
        if (!rc) out.insert(out.end(), color.begin(), color.end());
    } else if (spec.name == "gabor") {
        rc = gaborHist(bgr, out);
    } else if (spec.name == "dnn") {
        out = extra;
    } else if (spec.name == "mycustom") {
        rc = fourierHist(bgr, out);
        if (!rc) out.insert(out.end(), extra.begin(), extra.end());
    } else {
        err = "unknown feature '" + spec.name + "'";
        return false;
    }
    if (rc) {
        err = "image is empty, not 8-bit BGR, or too small for feature '" + spec.name + "'";
        return false;
    }
    return true;
}

int buildDatabase(const FeatureSpec& spec, const std::string& imageDir, const std::string& csvPath,
                  const cvc::FeatureTable* dnn, int limit, std::string& err) {
    std::vector<std::string> files = cvc::listImages(imageDir);
    if (files.empty()) {
        err = "no images found in '" + imageDir + "'";
        return -1;
    }
    if (limit > 0 && static_cast<size_t>(limit) < files.size()) files.resize(static_cast<size_t>(limit));
    int written = 0;
    for (const std::string& f : files) {
        const cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::fprintf(stderr, "skipping unreadable image %s\n", f.c_str());
            continue;
        }
        std::vector<float> feature;
        std::string e;
        if (!computeFeature(spec, img, f, dnn, feature, e)) {
            std::fprintf(stderr, "skipping %s: %s\n", f.c_str(), e.c_str());
            continue;
        }
        if (!cvc::appendFeatureRow(csvPath, cvc::baseName(f), feature, written == 0, &err)) return -1;
        written++;
    }
    if (written == 0) {
        err = "no feature could be computed";
        return -1;
    }
    return written;
}

std::vector<Match> rankMatches(const std::vector<float>& query, const cvc::FeatureTable& db,
                               Metric metric, size_t k, const std::string& excludeName, bool farthest) {
    const std::string exclude = excludeName.empty() ? "" : cvc::baseName(excludeName);
    std::vector<Match> all;
    all.reserve(db.size());
    for (size_t i = 0; i < db.size(); i++) {
        if (!exclude.empty() && cvc::baseName(db.names[i]) == exclude) continue;
        if (db.data[i].size() != query.size()) continue;
        all.push_back({i, db.names[i], distanceFor(metric, db.data[i], query)});
    }
    std::stable_sort(all.begin(), all.end(), [farthest](const Match& a, const Match& b) {
        return farthest ? a.distance > b.distance : a.distance < b.distance;
    });
    if (all.size() > k) all.resize(k);
    return all;
}

cv::Mat makeMontage(const cv::Mat& query, const std::string& queryName, const std::vector<cv::Mat>& images,
                    const std::vector<Match>& matches, int tw, int th) {
    const int cap = 22;
    const size_t n = 1 + images.size();
    cv::Mat out(th + cap, static_cast<int>(n) * tw, CV_8UC3, cv::Scalar(255, 255, 255));
    auto place = [&](const cv::Mat& img, size_t slot, const std::string& text, const cv::Scalar& color) {
        if (!img.empty()) {
            cv::Mat tile;
            cv::resize(img, tile, cv::Size(tw - 4, th - 4), 0, 0, cv::INTER_AREA);
            tile.copyTo(out(cv::Rect(static_cast<int>(slot) * tw + 2, 2, tw - 4, th - 4)));
        }
        cv::putText(out, text, cv::Point(static_cast<int>(slot) * tw + 4, th + 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
    };
    place(query, 0, "query " + queryName, cv::Scalar(0, 0, 200));
    for (size_t i = 0; i < images.size(); i++) {
        char text[96];
        const Match* m = i < matches.size() ? &matches[i] : nullptr;
        std::snprintf(text, sizeof text, "%zu %s d=%.3g", i + 1, m ? m->name.c_str() : "?", m ? static_cast<double>(m->distance) : 0.0);
        place(images[i], i + 1, text, cv::Scalar(0, 0, 0));
    }
    return out;
}
