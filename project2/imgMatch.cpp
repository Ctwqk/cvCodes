/*
  Find the images of a database that are most similar to a target image.

    imgMatch <target image> <csv dir> <feature> <N>        (original argument order)
    imgMatch <target image> <feature> [--top N] [options]

  options:
    --csv-dir DIR   feature databases (default project2/featureCsvs in the repository)
    --images DIR    where the database images live (default project2/images)
    --top N         number of matches (default 5)
    --worst         list the N least similar images instead
    --out DIR       write the result montage to DIR/match_<feature>_<target>.png
    --headless      never open windows (automatic when no display is available)

  Databases are created with establishCsv. If the target image is part of the
  database it is not reported as its own match.
*/
#include <cstdio>
#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "app_io.h"
#include "csv_util.h"
#include "retrieval.h"

namespace {
void usage(const char* prog) {
    std::printf("usage: %s <target image> <csv dir> <feature> <N>\n       %s <target image> <feature> [--top N] [--csv-dir DIR] [--images DIR] [--worst] [--out DIR]\nfeatures:",
                prog, prog);
    for (const FeatureSpec& s : featureSpecs()) std::printf(" %s", s.name.c_str());
    std::printf("\n");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"help", "worst", "headless"});
        const auto& pos = args.positional();
        if (args.has("help") || (pos.size() != 2 && pos.size() != 4)) {
            usage(argv[0]);
            return args.has("help") ? 0 : 1;
        }
        const std::string targetPath = pos[0];
        std::string csvDir = args.get("csv-dir", cvc::repoPath("project2/featureCsvs"));
        std::string featureName = pos[1];
        int top = args.getInt("top", 5);
        if (pos.size() == 4) {
            csvDir = pos[1];
            featureName = pos[2];
            top = std::stoi(pos[3]);
        }
        const std::string imageDir = args.get("images", cvc::repoPath("project2/images"));
        const FeatureSpec* spec = findFeatureSpec(featureName);
        if (!spec) {
            std::fprintf(stderr, "error: unknown feature '%s'\n", featureName.c_str());
            usage(argv[0]);
            return 1;
        }
        if (top < 1) {
            std::fprintf(stderr, "error: N must be at least 1\n");
            return 1;
        }

        const cv::Mat target = cv::imread(targetPath, cv::IMREAD_COLOR);
        if (target.empty()) {
            std::fprintf(stderr, "error: unable to read target image '%s'\n", targetPath.c_str());
            return 2;
        }

        std::string err;
        cvc::FeatureTable dnn, db;
        const bool needDnn = spec->needsDnn;
        if (needDnn && !cvc::readFeatureTable(csvDir + "/" + kDnnCsvName, dnn, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        const std::string dbPath = csvDir + "/" + spec->dbFile;
        if (!cvc::readFeatureTable(dbPath, db, &err)) {
            std::fprintf(stderr, "error: %s\n(create it with: establishCsv %s)\n", err.c_str(), spec->name.c_str());
            return 2;
        }

        std::vector<float> query;
        if (!computeFeature(*spec, target, targetPath, needDnn ? &dnn : nullptr, query, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 2;
        }
        const std::vector<Match> matches = rankMatches(query, db, spec->metric, static_cast<size_t>(top),
                                                       targetPath, args.has("worst"));
        if (matches.empty()) {
            std::fprintf(stderr, "error: no comparable rows in %s (feature length %zu)\n", dbPath.c_str(), query.size());
            return 2;
        }

        std::printf("%s matches for %s using '%s' (%zu images in the database):\n",
                    args.has("worst") ? "least similar" : "closest", targetPath.c_str(), spec->name.c_str(), db.size());
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < matches.size(); i++) {
            const std::string path = imageDir + "/" + cvc::baseName(matches[i].name);
            std::printf("%3zu  %-16s distance %.5g\n", i + 1, matches[i].name.c_str(), static_cast<double>(matches[i].distance));
            images.push_back(cv::imread(path, cv::IMREAD_COLOR));
            if (images.back().empty()) std::fprintf(stderr, "warning: cannot read database image %s\n", path.c_str());
        }

        const cv::Mat montage = makeMontage(target, cvc::baseName(targetPath), images, matches);
        cvc::Viewer viewer = cvc::Viewer::fromArgs(args);
        if (args.has("out")) {
            const std::string stem = std::filesystem::path(targetPath).stem().string();
            const std::string out = args.get("out") + "/match_" + spec->name + "_" + stem + (args.has("worst") ? "_worst" : "") + ".png";
            if (!cvc::writeImage(out, montage)) {
                std::fprintf(stderr, "error: cannot write %s\n", out.c_str());
                return 2;
            }
            std::printf("montage written to %s\n", out.c_str());
        }
        if (!viewer.headless()) {
            cv::imshow("matches", montage);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
