// Camera calibration, pose estimation and augmented-reality helpers (CS 5330 project 4).
//
// The pure parts (point layout, intrinsics files, calibration, Harris corners,
// board replacement, synthetic board rendering) live here so they can be unit
// tested; the apps in this directory are thin loops around them.
#ifndef AR_UTILS_H
#define AR_UTILS_H

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace p4 {

// Inner corners of the printed chessboard (columns x rows) and the camera model.
struct Intrinsics {
    cv::Size pattern{9, 6};
    cv::Mat cameraMatrix;              // 3x3 CV_64F
    std::vector<double> distCoeffs;    // k1 k2 p1 p2 [k3 ...]
};

// File format (compatible with the original project):
//   <columns> <rows>
//   <9 numbers: the camera matrix, row-major>
//   <distortion coefficients ...>
bool readIntrinsics(const std::string& path, Intrinsics& out, std::string* err = nullptr);
bool writeIntrinsics(const std::string& path, const Intrinsics& in, std::string* err = nullptr);

// World coordinates of the inner corners in the order OpenCV reports them:
// (j, -i, 0) for column j and row i, one unit per chessboard square. The board lies in the z = 0
// plane with x to the right, y up and z pointing towards the viewer.
std::vector<cv::Point3f> chessboardPoints(cv::Size pattern);

// Detect the inner corners in a greyscale image and refine them to sub-pixel accuracy.
bool findBoardCorners(const cv::Mat& gray, cv::Size pattern, std::vector<cv::Point2f>& corners);

struct CalibrationResult {
    cv::Mat cameraMatrix;
    std::vector<double> distCoeffs;
    double rms = 0;          // RMS reprojection error reported by cv::calibrateCamera (pixels)
    int views = 0;
};

// cv::calibrateCamera over several views of the board. `flags` default to CALIB_FIX_ASPECT_RATIO as in the
// original (square pixels). Needs at least 3 views; returns false otherwise.
bool calibrateFromCorners(cv::Size imageSize, cv::Size pattern,
                          const std::vector<std::vector<cv::Point2f>>& corners, CalibrationResult& out,
                          int flags = cv::CALIB_FIX_ASPECT_RATIO);

// Pose of the board (rotation vector and translation, in board units) from its detected corners.
bool estimatePose(const Intrinsics& cam, const std::vector<cv::Point2f>& corners, cv::Mat& rvec, cv::Mat& tvec);

// Mean distance (pixels) between detected corners and the board projected with the given pose.
double reprojectionError(const Intrinsics& cam, const std::vector<cv::Point2f>& corners, const cv::Mat& rvec,
                         const cv::Mat& tvec);

// Three coloured axes floating 3 units above the board's first corner.
void drawAxes(cv::Mat& frame, const Intrinsics& cam, const cv::Mat& rvec, const cv::Mat& tvec);
// A virtual 5-point star (double pyramid) above the board.
void drawStar(cv::Mat& frame, const Intrinsics& cam, const cv::Mat& rvec, const cv::Mat& tvec);
// Pose read-out ("rot:" / "trs:") in the top-left corner.
void drawPoseText(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec);

// Harris corner response of a greyscale image, normalised to [0, 255]; the pixels above `threshold` are
// returned. With `nonMaxSuppression` only local maxima (3x3) are kept, one point per corner.
std::vector<cv::Point2f> harrisCorners(const cv::Mat& gray, double threshold = 160, bool nonMaxSuppression = true);

// Paste `replacement` over the board: it is warped onto the quadrilateral spanned by the four outer inner
// corners and everything inside that quadrilateral is replaced. Returns false if the corner list is not
// pattern.area() long. The corners are also blacked out in `grayForSearch` (optional) so the next board in
// the frame can be found.
bool replaceBoard(cv::Mat& frame, const cv::Mat& replacement, const std::vector<cv::Point2f>& corners,
                  cv::Size pattern, cv::Mat* grayForSearch = nullptr);

// ---- synthetic views of the chessboard ------------------------------------------------------------------

// The printed pattern image (checkerboard.png) together with the homography that maps board coordinates
// (j, -i) to its pixels.
struct BoardTemplate {
    cv::Mat image;               // BGR
    cv::Mat boardToImage;        // 3x3 CV_64F
    cv::Size pattern{9, 6};
};
bool loadBoardTemplate(const std::string& path, cv::Size pattern, BoardTemplate& out, std::string* err = nullptr);

// Render what the camera `cam` sees when the board has pose (rvec, tvec): the template is mapped through the
// pinhole model and the lens distortion, on a grey table with sensor noise. `seed` makes the noise reproducible.
cv::Mat renderBoardView(const BoardTemplate& board, const Intrinsics& cam, const cv::Mat& rvec,
                        const cv::Mat& tvec, cv::Size imageSize, unsigned seed = 0);

// A pose for view `index` of `count`: the board is tilted up to ~40 degrees, rotated in plane and moved around,
// always inside the frame. Deterministic for a given seed.
void samplePose(int index, int count, unsigned seed, cv::Mat& rvec, cv::Mat& tvec);

// A smooth fly-by for t in [0, 1] (tilt, in-plane rotation and distance all vary): used for demo videos.
void trajectoryPose(double t, cv::Mat& rvec, cv::Mat& tvec);

}  // namespace p4

#endif
