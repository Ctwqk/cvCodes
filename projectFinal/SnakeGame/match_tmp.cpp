/* Taiwei Cui, April 2024: SIFT pattern acquisition + Lucas-Kanade tracking.
 * Corrected frame order, correspondence indices and perspective division.
 */
#include "saveOrReadPattern.hpp"
#include "patternMatch.hpp"
#include "app_io.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    try {
        const cvc::Args args(argc, argv, {"headless", "help"});
        if (args.has("help")) {
            cout << "usage: matchPatterns [--patterns FILE] [--source 0|video|image] [--out DIR] [--max-frames N] [--headless]\n";
            return 0;
        }
        vector<vector<KeyPoint>> patterns;
        vector<Mat> descriptors, corners;
        vector<vector<Point3f>> world;
        if (readPatternsFromCsv(args.get("patterns", "results/patterns.txt"), patterns, descriptors, corners, world))
            throw runtime_error("cannot read patterns; run buildPatterns first");
        auto cap = cvc::openSource(args.get("source", "0"));
        auto viewer = cvc::Viewer::fromArgs(args, "results/tracking");
        auto detector = SIFT::create(1000);
        BFMatcher matcher(NORM_L2);
        Mat frame, gray, previous, H;
        vector<Point2f> tracked, reference;
        int target = -1;
        const int limit = args.getInt("max-frames", -1);
        for (int n = 0; limit < 0 || n < limit; ++n) {
            cap >> frame;
            if (frame.empty()) break;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            bool found = false;
            if (tracked.size() >= 4 && !previous.empty()) {
                vector<Point2f> next, good, patternGood;
                vector<uchar> status;
                vector<float> error;
                calcOpticalFlowPyrLK(previous, gray, tracked, next, status, error);
                for (size_t i = 0; i < next.size(); ++i) {
                    if (status[i] && error[i] < 10) {
                        good.push_back(next[i]);
                        patternGood.push_back(reference[i]);
                    }
                }
                tracked = std::move(good);
                reference = std::move(patternGood);
                found = getHomoMat(reference, tracked, H) == 0;
            }
            if (!found) {
                vector<KeyPoint> keypoints;
                Mat desc;
                detector->detectAndCompute(gray, noArray(), keypoints, desc);
                vector<DMatch> matches;
                found = matchPattern(descriptors, desc, patterns, keypoints, matcher, matches, target, H, -1) == 0;
                tracked.clear();
                reference.clear();
                if (found) for (const auto& m : matches) {
                    tracked.push_back(keypoints[m.queryIdx].pt);
                    reference.push_back(patterns[target][m.trainIdx].pt);
                }
            }
            if (found) {
                vector<Point2f> box, projected;
                for (int i : {0, 1, 3, 2}) box.emplace_back(corners[target].at<double>(0, i), corners[target].at<double>(1, i));
                perspectiveTransform(box, projected, H);
                for (size_t i = 0; i < projected.size(); ++i) line(frame, projected[i], projected[(i+1)%4], Scalar(0,255,0), 3, LINE_AA);
                for (const auto& point : tracked) circle(frame, point, 2, Scalar(0,200,255), -1);
            }
            const string label = found ? "pattern " + to_string(target) + ": " + to_string(tracked.size()) + " points" : "no pattern";
            putText(frame, label, Point(12,28), FONT_HERSHEY_SIMPLEX, .65, Scalar(0,0,255), 2);
            cout << "frame " << n << ": " << label << '\n';
            viewer.show("tracking", frame);
            gray.copyTo(previous);
            if (viewer.waitKey(5) == 'q') break;
        }
        return 0;
    } catch (const exception& e) {
        cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
