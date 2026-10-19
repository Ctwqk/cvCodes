/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing boxes
  (course skeleton, adapted: configurable cascade path, no exit() on failure).
*/
#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// File name of the Haar cascade shipped with the repository.
#define FACE_CASCADE_FILE_NAME "haarcascade_frontalface_alt2.xml"

// Locates the Haar cascade. Search order: $CVC_FACE_CASCADE, the copy in this
// repository (project1/task2/FaceDetect), the current directory, OpenCV's
// share directory. Returns "" when none exists.
std::string findFaceCascade();

// Detect faces in a greyscale image (processed at half size for speed).
// Returns 0 on success; -1 if the cascade cannot be loaded (faces is cleared).
int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces);

// Draw boxes around faces at least `minWidth` wide; `scale` maps face
// coordinates into `frame` coordinates.
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth = 50, float scale = 1.0);

#endif
