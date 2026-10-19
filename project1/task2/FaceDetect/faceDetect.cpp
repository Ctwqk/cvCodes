/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them.
*/
#include "faceDetect.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "app_io.h"

std::string findFaceCascade() {
    namespace fs = std::filesystem;
    std::vector<std::string> candidates;
    if (const char* env = std::getenv("CVC_FACE_CASCADE")) candidates.push_back(env);
    candidates.push_back(cvc::repoPath("project1/task2/FaceDetect/" FACE_CASCADE_FILE_NAME));
    candidates.push_back("./" FACE_CASCADE_FILE_NAME);
    candidates.push_back("/usr/share/opencv4/haarcascades/" FACE_CASCADE_FILE_NAME);
    for (const std::string& c : candidates) {
        std::error_code ec;
        if (!c.empty() && fs::is_regular_file(c, ec)) return c;
    }
    return "";
}

/*
  Arguments:
  cv::Mat grey  - a greyscale source image in which to detect faces
  std::vector<cv::Rect> &faces - rectangles indicating where faces were found
     if the length of the vector is zero, no faces were found
 */
int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces) {
    static cv::Mat half;
    static cv::CascadeClassifier face_cascade;
    static bool triedLoading = false;

    faces.clear();
    if (face_cascade.empty()) {
        if (!triedLoading) {
            triedLoading = true;
            const std::string path = findFaceCascade();
            if (path.empty() || !face_cascade.load(path)) {
                std::fprintf(stderr,
                             "Unable to load face cascade '%s' (set CVC_FACE_CASCADE to its path)\n",
                             FACE_CASCADE_FILE_NAME);
            }
        }
        if (face_cascade.empty()) return -1;
    }

    // cut the image size in half to reduce processing time
    cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));
    cv::equalizeHist(half, half);
    face_cascade.detectMultiScale(half, faces);

    // adjust the rectangle sizes back to the full size image
    for (cv::Rect& f : faces) {
        f.x *= 2;
        f.y *= 2;
        f.width *= 2;
        f.height *= 2;
    }
    return 0;
}

/* Draws rectangles into frame given a vector of rectangles.
   minWidth: ignore rectangles with a width smaller than this
   scale:    scale the rectangle values by this factor */
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth, float scale) {
    const cv::Scalar wcolor(170, 120, 110);  // (B, G, R)
    for (const cv::Rect& f : faces) {
        if (f.width > minWidth) {
            cv::Rect face(f);
            face.x = static_cast<int>(face.x * scale);
            face.y = static_cast<int>(face.y * scale);
            face.width = static_cast<int>(face.width * scale);
            face.height = static_cast<int>(face.height * scale);
            cv::rectangle(frame, face, wcolor, 3);
        }
    }
    return 0;
}
