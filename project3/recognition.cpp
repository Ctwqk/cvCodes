#include "recognition.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>

#include "distance.h"

std::string labelOf(const std::string& rowName) {
    const size_t pos = rowName.find(':');
    return pos == std::string::npos ? rowName : rowName.substr(0, pos);
}

std::string labelFromFilename(const std::string& path) {
    std::string name = cvc::baseName(path);
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    while (!name.empty() && std::isdigit(static_cast<unsigned char>(name.back()))) name.pop_back();
    return name;
}

std::vector<float> columnStdDev(const cvc::FeatureTable& db, long skipRow) {
    if (db.data.empty()) return {};
    return columnStdDev(db, skipRow, db.data[0].size());
}

std::vector<float> columnStdDev(const cvc::FeatureTable& db, long skipRow, size_t d) {
    std::vector<double> mean(d, 0.0), var(d, 0.0);
    size_t n = 0;
    for (size_t r = 0; r < db.data.size(); r++) {
        const auto& row = db.data[r];
        if (row.size() != d || static_cast<long>(r) == skipRow) continue;
        for (size_t j = 0; j < d; j++) mean[j] += row[j];
        n++;
    }
    if (n == 0) return {};
    for (double& m : mean) m /= static_cast<double>(n);
    for (size_t r = 0; r < db.data.size(); r++) {
        const auto& row = db.data[r];
        if (row.size() != d || static_cast<long>(r) == skipRow) continue;
        for (size_t j = 0; j < d; j++) var[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
    }
    std::vector<float> sd(d);
    for (size_t j = 0; j < d; j++) sd[j] = static_cast<float>(std::sqrt(var[j] / static_cast<double>(n)));
    return sd;
}

Classification classify(const std::vector<float>& feature, const cvc::FeatureTable& db,
                        const ClassifierConfig& config, const std::vector<float>& scale, long skipRow) {
    Classification result;
    std::vector<std::pair<float, size_t>> ranked;
    const std::vector<float> sd = (config.kind == DistanceKind::ScaledEuclidean && scale.empty())
                                      ? columnStdDev(db, skipRow, feature.size())
                                      : scale;
    for (size_t i = 0; i < db.size(); i++) {
        if (static_cast<long>(i) == skipRow || db.data[i].size() != feature.size()) continue;
        float d = 0;
        switch (config.kind) {
            case DistanceKind::Cosine: d = cvc::cosineDistance(feature, db.data[i]); break;
            case DistanceKind::Euclidean: d = cvc::euclidean(feature, db.data[i]); break;
            case DistanceKind::ScaledEuclidean: d = cvc::scaledEuclidean(feature, db.data[i], sd); break;
        }
        ranked.emplace_back(d, i);
    }
    if (ranked.empty()) return result;
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    const size_t k = static_cast<size_t>(std::max(1, config.k));
    if (ranked.size() > k) ranked.resize(k);
    std::map<std::string, int> votes;
    for (const auto& [d, i] : ranked) {
        votes[labelOf(db.names[i])]++;
        result.neighbours.push_back(i);
    }
    result.distance = ranked.front().first;
    // most votes wins; among equal counts the class whose neighbour is closest (iteration order of `ranked`)
    int best = 0;
    for (const auto& [d, i] : ranked) {
        const std::string label = labelOf(db.names[i]);
        if (votes[label] > best) {
            best = votes[label];
            result.label = label;
        }
    }
    return result;
}

int ConfusionMatrix::total() const {
    int t = 0;
    for (const auto& row : counts)
        for (int c : row) t += c;
    return t;
}

int ConfusionMatrix::correct() const {
    int t = 0;
    for (size_t i = 0; i < counts.size(); i++) t += counts[i][i];
    return t;
}

double ConfusionMatrix::accuracy() const { return total() ? static_cast<double>(correct()) / total() : 0.0; }

ConfusionMatrix leaveOneOut(const cvc::FeatureTable& db, const ClassifierConfig& config) {
    ConfusionMatrix cm;
    std::set<std::string> names;
    for (const std::string& n : db.names) names.insert(labelOf(n));
    cm.labels.assign(names.begin(), names.end());
    cm.counts.assign(cm.labels.size(), std::vector<int>(cm.labels.size(), 0));
    for (size_t i = 0; i < db.size(); i++) {
        const Classification c = classify(db.data[i], db, config, {}, static_cast<long>(i));
        if (c.label.empty()) continue;
        const size_t truth = static_cast<size_t>(std::find(cm.labels.begin(), cm.labels.end(), labelOf(db.names[i])) - cm.labels.begin());
        const size_t pred = static_cast<size_t>(std::find(cm.labels.begin(), cm.labels.end(), c.label) - cm.labels.begin());
        cm.counts[truth][pred]++;
    }
    return cm;
}
