#include "ar_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace p4 {

// ---- intrinsics file --------------------------------------------------------------------------------

bool readIntrinsics(const std::string& path, Intrinsics& out, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "failed to read from file '" + path + "'";
        return false;
    }
    int x = 0, y = 0;
    if (!(in >> x >> y) || x < 2 || y < 2) {
        if (err) *err = path + ": expected the pattern size (columns rows) on the first line";
        return false;
    }
    cv::Mat K(3, 3, CV_64F);
    double* p = K.ptr<double>(0);
    for (int i = 0; i < 9; i++) {
        if (!(in >> p[i])) {
            if (err) *err = path + ": expected nine camera matrix values";
            return false;
        }
    }
    std::vector<double> dist;
    double v;
    while (in >> v) dist.push_back(v);
    out.pattern = cv::Size(x, y);
    out.cameraMatrix = K;
    out.distCoeffs = dist;
    return true;
}

bool writeIntrinsics(const std::string& path, const Intrinsics& in, std::string* err) {
    std::ofstream out(path);
    if (!out || in.cameraMatrix.rows != 3 || in.cameraMatrix.cols != 3) {
        if (err) *err = "failed to write to file '" + path + "'";
        return false;
    }
    out.precision(10);
    out << in.pattern.width << " " << in.pattern.height << "\n";
    cv::Mat K;
    in.cameraMatrix.convertTo(K, CV_64F);
    const double* p = K.ptr<double>(0);
    for (int i = 0; i < 9; i++) out << p[i] << " ";
    out << "\n";
    for (double d : in.distCoeffs) out << d << " ";
    out << "\n";
    return static_cast<bool>(out);
}

// ---- calibration -------------------------------------------------------------------------------------

std::vector<cv::Point3f> chessboardPoints(cv::Size pattern) {
    std::vector<cv::Point3f> pts;
    for (int i = 0; i < pattern.height; i++)
        for (int j = 0; j < pattern.width; j++) pts.emplace_back(static_cast<float>(j), static_cast<float>(-i), 0.f);
    return pts;
}

bool findBoardCorners(const cv::Mat& gray, cv::Size pattern, std::vector<cv::Point2f>& corners) {
    if (gray.empty() || gray.type() != CV_8UC1) return false;
    bool found = false;
    try {
        found = cv::findChessboardCorners(gray, pattern, corners);
    } catch (const cv::Exception&) {  // degenerate (tiny) images trip assertions inside OpenCV
        found = false;
    }
    if (!found) {
        corners.clear();
        return false;
    }
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    return true;
}

bool calibrateFromCorners(cv::Size imageSize, cv::Size pattern,
                          const std::vector<std::vector<cv::Point2f>>& corners, CalibrationResult& out, int flags) {
    if (corners.size() < 3) return false;
    const std::vector<std::vector<cv::Point3f>> objects(corners.size(), chessboardPoints(pattern));
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F), dist;
    std::vector<cv::Mat> rvecs, tvecs;
    out.rms = cv::calibrateCamera(objects, corners, imageSize, K, dist, rvecs, tvecs, flags);
    out.cameraMatrix = K;
    out.distCoeffs.assign(dist.ptr<double>(0), dist.ptr<double>(0) + dist.total());
    out.views = static_cast<int>(corners.size());
    return true;
}

bool estimatePose(const Intrinsics& cam, const std::vector<cv::Point2f>& corners, cv::Mat& rvec, cv::Mat& tvec) {
    if (static_cast<int>(corners.size()) != cam.pattern.area()) return false;
    return cv::solvePnP(chessboardPoints(cam.pattern), corners, cam.cameraMatrix, cam.distCoeffs, rvec, tvec);
}

double reprojectionError(const Intrinsics& cam, const std::vector<cv::Point2f>& corners, const cv::Mat& rvec,
                         const cv::Mat& tvec) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(chessboardPoints(cam.pattern), rvec, tvec, cam.cameraMatrix, cam.distCoeffs, projected);
    double sum = 0;
    for (size_t i = 0; i < corners.size() && i < projected.size(); i++) sum += cv::norm(corners[i] - projected[i]);
    return corners.empty() ? 0.0 : sum / static_cast<double>(corners.size());
}

// ---- drawing -----------------------------------------------------------------------------------------

void drawAxes(cv::Mat& frame, const Intrinsics& cam, const cv::Mat& rvec, const cv::Mat& tvec) {
    const std::vector<cv::Point3f> targets = {{0, 0, 3}, {0, 1, 3}, {0, 0, 4}, {1, 0, 3}};
    const cv::Scalar colours[3] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
    std::vector<cv::Point2f> r;
    cv::projectPoints(targets, rvec, tvec, cam.cameraMatrix, cam.distCoeffs, r);
    cv::circle(frame, r[0], 5, cv::Scalar(255, 0, 255), -1);
    for (int i = 1; i < 4; i++) cv::line(frame, r[0], r[i], colours[i - 1], 3, cv::LINE_4);
}

void drawStar(cv::Mat& frame, const Intrinsics& cam, const cv::Mat& rvec, const cv::Mat& tvec) {
    // the five points of a pentagram at height 1, plus an apex above and below its centre
    const std::vector<cv::Point3f> star = {{4, 0, 1},       {6.8532f, -1.9271f, 1}, {5.7642f, -5.4271f, 1},
                                           {2.2366f, -5.4271f, 1}, {1.1468f, -1.9271f, 1}, {4, -3, 2},
                                           {4, -3, 0}};
    std::vector<cv::Point2f> r;
    cv::projectPoints(star, rvec, tvec, cam.cameraMatrix, cam.distCoeffs, r);
    for (int i = 0; i < 5; i++) {
        cv::line(frame, r[i], r[5], cv::Scalar(0, 0, 255), 3, cv::LINE_4);
        cv::line(frame, r[i], r[6], cv::Scalar(0, 0, 255), 3, cv::LINE_4);
        for (int j = i + 2; j < 5; j++) {
            if (i == 0 && j == 4) continue;
            cv::line(frame, r[i], r[j], cv::Scalar(0, 255, 255), 3, cv::LINE_4);
        }
    }
}

void drawPoseText(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat r, t;
    rvec.convertTo(r, CV_64F);
    tvec.convertTo(t, CV_64F);
    cv::putText(frame, "rot:", cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(frame, "trs:", cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    for (int i = 0; i < 3; i++) {
        cv::putText(frame, std::to_string(static_cast<float>(r.at<double>(i))), cv::Point(35, 20 + i * 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, std::to_string(static_cast<float>(t.at<double>(i))), cv::Point(35, 80 + i * 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
}

// ---- Harris corners -----------------------------------------------------------------------------------

std::vector<cv::Point2f> harrisCorners(const cv::Mat& gray, double threshold, bool nonMaxSuppression) {
    std::vector<cv::Point2f> points;
    if (gray.empty() || gray.type() != CV_8UC1) return points;
    cv::Mat response;
    cv::cornerHarris(gray, response, 3, 3, 0.05, cv::BORDER_DEFAULT);
    cv::normalize(response, response, 0, 255, cv::NORM_MINMAX, CV_32F);
    cv::Mat localMax;
    if (nonMaxSuppression) cv::dilate(response, localMax, cv::Mat());  // 3x3
    for (int i = 0; i < response.rows; i++) {
        const float* p = response.ptr<float>(i);
        const float* m = nonMaxSuppression ? localMax.ptr<float>(i) : nullptr;
        for (int j = 0; j < response.cols; j++) {
            if (p[j] > threshold && (!m || p[j] >= m[j])) points.emplace_back(static_cast<float>(j), static_cast<float>(i));
        }
    }
    return points;
}

// ---- replace the board by a picture --------------------------------------------------------------------

bool replaceBoard(cv::Mat& frame, const cv::Mat& replacement, const std::vector<cv::Point2f>& corners,
                  cv::Size pattern, cv::Mat* grayForSearch) {
    if (static_cast<int>(corners.size()) != pattern.area() || frame.empty() || replacement.empty()) return false;
    // homography from grid coordinates (column, row) to the image, then push the outer corners one
    // square outwards so that the whole printed board (not only the inner corners) is covered
    std::vector<cv::Point2f> grid;
    for (int i = 0; i < pattern.height; i++)
        for (int j = 0; j < pattern.width; j++) grid.emplace_back(static_cast<float>(j), static_cast<float>(i));
    const cv::Mat H = cv::findHomography(grid, corners);
    if (H.empty()) return false;
    const float W = static_cast<float>(pattern.width - 1), Hh = static_cast<float>(pattern.height - 1);
    const float m = 1.1f;  // a bit more than one square: lens distortion bends the fitted homography
    const std::vector<cv::Point2f> outer = {{-m, -m}, {W + m, -m}, {W + m, Hh + m}, {-m, Hh + m}};
    std::vector<cv::Point2f> quad;
    cv::perspectiveTransform(outer, quad, H);

    std::vector<cv::Point> poly;
    for (const cv::Point2f& p : quad) poly.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));

    const std::vector<cv::Point2f> src = {{0, 0},
                                          {static_cast<float>(replacement.cols), 0},
                                          {static_cast<float>(replacement.cols), static_cast<float>(replacement.rows)},
                                          {0, static_cast<float>(replacement.rows)}};
    cv::Mat warped;
    cv::warpPerspective(replacement, warped, cv::getPerspectiveTransform(src, quad), frame.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT);
    if (warped.channels() != frame.channels()) {
        cv::cvtColor(warped, warped, warped.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
    }
    warped.copyTo(frame, mask);
    if (grayForSearch && !grayForSearch->empty()) cv::fillConvexPoly(*grayForSearch, poly, cv::Scalar(0));
    return true;
}

// ---- synthetic views -----------------------------------------------------------------------------------

bool loadBoardTemplate(const std::string& path, cv::Size pattern, BoardTemplate& out, std::string* err) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        if (err) *err = "cannot read board image '" + path + "'";
        return false;
    }
    // detect on a reduced copy (fast and just as reliable on a clean print), refine on the full image
    const double shrink = std::max(1.0, std::max(img.cols, img.rows) / 600.0);
    cv::Mat gray, small;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, small, cv::Size(), 1.0 / shrink, 1.0 / shrink, cv::INTER_AREA);
    std::vector<cv::Point2f> corners;
    if (!cv::findChessboardCorners(small, pattern, corners)) {
        if (err) *err = "no " + std::to_string(pattern.width) + "x" + std::to_string(pattern.height) + " chessboard in '" + path + "'";
        return false;
    }
    for (cv::Point2f& c : corners) c *= static_cast<float>(shrink);
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
    std::vector<cv::Point2f> board;
    for (const cv::Point3f& p : chessboardPoints(pattern)) board.emplace_back(p.x, p.y);
    out.image = img;
    out.pattern = pattern;
    out.boardToImage = cv::findHomography(board, corners);
    return !out.boardToImage.empty();
}

cv::Mat renderBoardView(const BoardTemplate& board, const Intrinsics& cam, const cv::Mat& rvec,
                        const cv::Mat& tvec, cv::Size size, unsigned seed) {
    // board plane (X, Y, 1) -> pinhole pixel is G = K [r1 r2 t]
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat t;
    tvec.convertTo(t, CV_64F);
    cv::Mat RT(3, 3, CV_64F);
    R.col(0).copyTo(RT.col(0));
    R.col(1).copyTo(RT.col(1));
    t.reshape(1, 3).copyTo(RT.col(2));
    const cv::Mat G = cam.cameraMatrix * RT;
    const cv::Mat toTemplate = board.boardToImage * G.inv();

    // for every pixel of the (distorted) output find the matching pinhole pixel, then the template pixel
    std::vector<cv::Point2f> distorted;
    distorted.reserve(static_cast<size_t>(size.area()));
    for (int v = 0; v < size.height; v++)
        for (int u = 0; u < size.width; u++) distorted.emplace_back(static_cast<float>(u), static_cast<float>(v));
    std::vector<cv::Point2f> pinhole, source;
    if (cam.distCoeffs.empty()) {
        pinhole = distorted;
    } else {
        cv::undistortPoints(distorted, pinhole, cam.cameraMatrix, cam.distCoeffs, cv::noArray(), cam.cameraMatrix);
    }
    cv::perspectiveTransform(pinhole, source, toTemplate);
    cv::Mat mapX(size, CV_32F), mapY(size, CV_32F);
    for (int v = 0; v < size.height; v++)
        for (int u = 0; u < size.width; u++) {
            const cv::Point2f& s = source[static_cast<size_t>(v) * size.width + u];
            mapX.at<float>(v, u) = s.x;
            mapY.at<float>(v, u) = s.y;
        }
    cv::Mat view;
    cv::remap(board.image, view, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(105, 120, 135));

    // a little optics and sensor noise
    cv::GaussianBlur(view, view, cv::Size(3, 3), 0.7);
    cv::Mat noise(view.size(), view.type());
    cv::RNG rng(seed);
    rng.fill(noise, cv::RNG::NORMAL, 0, 2.0);
    cv::add(view, noise, view);
    return view;
}

namespace {
void composePose(double alpha, double beta, double gamma, double dx, double dy, double z, cv::Mat& rvec, cv::Mat& tvec) {
    cv::Mat Rx, Ry, Rz;
    cv::Rodrigues(cv::Vec3d(alpha, 0, 0), Rx);
    cv::Rodrigues(cv::Vec3d(0, beta, 0), Ry);
    cv::Rodrigues(cv::Vec3d(0, 0, gamma), Rz);
    // The board has x to the right, y up and z towards the viewer, the camera has x right, y down and z
    // forward: a board facing the camera is the camera frame turned by 180 degrees about x.
    const cv::Mat flip = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, -1);
    const cv::Mat R = Rz * Rx * Ry * flip;
    cv::Rodrigues(R, rvec);
    // board centre (4, -2.5, 0) lands at (dx, dy, z) in camera coordinates
    const cv::Mat centre = (cv::Mat_<double>(3, 1) << 4.0, -2.5, 0.0);
    const cv::Mat target = (cv::Mat_<double>(3, 1) << dx, dy, z);
    tvec = target - R * centre;
}
}  // namespace

void samplePose(int index, int count, unsigned seed, cv::Mat& rvec, cv::Mat& tvec) {
    (void)count;
    cv::RNG rng(seed * 7919u + static_cast<unsigned>(index) * 104729u + 1u);
    const double deg = CV_PI / 180.0;
    const double alpha = rng.uniform(-40.0, 40.0) * deg, beta = rng.uniform(-40.0, 40.0) * deg;
    const double gamma = rng.uniform(-30.0, 30.0) * deg;
    const double z = rng.uniform(12.5, 18.0);
    const double dx = rng.uniform(-1.0, 1.0) * 0.7 * std::max(0.0, 320.0 * z / 520.0 - 6.0);
    const double dy = rng.uniform(-1.0, 1.0) * 0.7 * std::max(0.0, 240.0 * z / 520.0 - 5.4);
    composePose(alpha, beta, gamma, dx, dy, z, rvec, tvec);
}

void trajectoryPose(double t, cv::Mat& rvec, cv::Mat& tvec) {
    const double deg = CV_PI / 180.0, w = 2 * CV_PI * t;
    composePose(28.0 * deg * std::sin(w), 32.0 * deg * std::sin(1.5 * w + 1.0), 22.0 * deg * std::sin(0.5 * w),
                1.0 * std::sin(w), 0.5 * std::cos(1.3 * w), 15.0 + 2.0 * std::sin(2.0 * w), rvec, tvec);
}

}  // namespace p4
