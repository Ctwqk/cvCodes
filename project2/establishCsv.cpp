/*
  Build the feature database (CSV) for one feature, or for all of them.

    establishCsv <image dir> <csv dir> <feature|all>        (original argument order)
    establishCsv <feature|all> [--images DIR] [--csv-dir DIR] [--limit N] [--dnn-csv FILE]

  Defaults: --images project2/images, --csv-dir project2/featureCsvs inside the
  repository. `all` builds every feature that does not need the ResNet18
  embeddings, plus `mycustom` when featureCsvs/ResNet18_olym.csv is present.
  Every row is "<image base name>,<feature values...>".
*/
#include <chrono>
#include <cstdio>

#include "app_io.h"
#include "csv_util.h"
#include "retrieval.h"

namespace {
void usage(const char* prog) {
    std::printf("usage: %s <image dir> <csv dir> <feature|all>\n       %s <feature|all> [--images DIR] [--csv-dir DIR] [--limit N] [--dnn-csv FILE]\nfeatures:\n",
                prog, prog);
    for (const FeatureSpec& s : featureSpecs()) {
        if (s.name != "dnn") std::printf("  %-13s %s\n", s.name.c_str(), s.description.c_str());
    }
    std::printf("  all           every feature above\n");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help"});
        const auto& pos = args.positional();
        if (args.has("help") || (pos.size() != 1 && pos.size() != 3)) {
            usage(argv[0]);
            return args.has("help") ? 0 : 1;
        }
        std::string imageDir = args.get("images", cvc::repoPath("project2/images"));
        std::string csvDir = args.get("csv-dir", cvc::repoPath("project2/featureCsvs"));
        std::string featureName = pos.back();
        if (pos.size() == 3) {
            imageDir = pos[0];
            csvDir = pos[1];
        }
        const int limit = args.getInt("limit", 0);
        const std::string dnnPath = args.get("dnn-csv", csvDir + "/" + kDnnCsvName);

        std::vector<const FeatureSpec*> todo;
        if (featureName == "all") {
            for (const FeatureSpec& s : featureSpecs()) {
                if (s.name != "dnn") todo.push_back(&s);
            }
        } else if (const FeatureSpec* s = findFeatureSpec(featureName)) {
            if (s->name == "dnn") {
                std::fprintf(stderr, "'dnn' is the provided %s; there is nothing to build\n", kDnnCsvName);
                return 1;
            }
            todo.push_back(s);
        } else {
            std::fprintf(stderr, "error: unknown feature '%s'\n", featureName.c_str());
            usage(argv[0]);
            return 1;
        }

        cvc::FeatureTable dnn;
        bool haveDnn = false;
        for (const FeatureSpec* s : todo) {
            if (s->needsDnn) {
                std::string err;
                haveDnn = cvc::readFeatureTable(dnnPath, dnn, &err);
                if (!haveDnn) std::fprintf(stderr, "note: %s\n", err.c_str());
                break;
            }
        }

        int failures = 0;
        for (const FeatureSpec* s : todo) {
            if (s->needsDnn && !haveDnn) {
                std::fprintf(stderr, "skipping '%s': needs %s\n", s->name.c_str(), dnnPath.c_str());
                failures++;
                continue;
            }
            const std::string csvPath = csvDir + "/" + s->dbFile;
            std::string err;
            const auto t0 = std::chrono::steady_clock::now();
            const int n = buildDatabase(*s, imageDir, csvPath, s->needsDnn ? &dnn : nullptr, limit, err);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (n < 0) {
                std::fprintf(stderr, "error building '%s': %s\n", s->name.c_str(), err.c_str());
                failures++;
                continue;
            }
            std::printf("%-13s %5d images in %6.1f s -> %s\n", s->name.c_str(), n, sec, csvPath.c_str());
        }
        return failures ? 2 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
