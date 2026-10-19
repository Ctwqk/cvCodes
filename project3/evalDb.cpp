/*
  Leave-one-out evaluation of a feature database: every row is classified against all the others.

  usage: evalDb [--db FILE ...] [--features shape|dnn] [--metric cosine|euclid|scaled] [--k N]
  With no --metric every metric is evaluated for k = 1 and k = 3, one table per run.
*/
#include <cstdio>

#include "app_io.h"
#include "csv_util.h"
#include "dbtools.h"

namespace {
void printMatrix(const ConfusionMatrix& cm) {
    std::printf("            ");
    for (const std::string& l : cm.labels) std::printf("%7s", l.c_str());
    std::printf("   (columns = predicted)\n");
    for (size_t i = 0; i < cm.labels.size(); i++) {
        std::printf("  %-10s", cm.labels[i].c_str());
        for (int c : cm.counts[i]) std::printf("%7d", c);
        std::printf("\n");
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help", "matrix"});
        if (args.has("help")) {
            std::printf("usage: %s [--db FILE] [--features shape|dnn] [--metric cosine|euclid|scaled] [--k N] [--matrix]\n", argv[0]);
            return 0;
        }
        const FeatureKind kind = parseFeatureKind(args.get("features", "shape"));
        const std::string dbPath = args.get("db", defaultDbPath(kind));
        cvc::FeatureTable db;
        std::string err;
        if (!cvc::readFeatureTable(dbPath, db, &err)) {
            std::fprintf(stderr, "error: %s\n(build it with buildDb)\n", err.c_str());
            return 2;
        }
        std::printf("%s: %zu rows\n", dbPath.c_str(), db.size());
        std::vector<std::string> metrics = {"cosine", "euclid", "scaled"};
        if (args.has("metric")) metrics = {args.get("metric")};
        std::vector<int> ks = {1, 3};
        if (args.has("k")) ks = {args.getInt("k", 1)};
        for (const std::string& m : metrics) {
            for (int k : ks) {
                ClassifierConfig cfg;
                cfg.kind = parseDistanceKind(m);
                cfg.k = k;
                const ConfusionMatrix cm = leaveOneOut(db, cfg);
                std::printf("metric %-7s k=%d  leave-one-out accuracy %5.1f%% (%d/%d)\n", m.c_str(), k,
                            100 * cm.accuracy(), cm.correct(), cm.total());
                if (args.has("matrix")) printMatrix(cm);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
