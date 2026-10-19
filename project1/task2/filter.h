/*
Author: Taiwei Cui
Time: 2024-1-22

Image filters written from scratch for CS 5330 project 1 (BGR, 8-bit input).

Conventions
 - Inputs are `const cv::Mat&`; outputs are written to `dst`, which may alias
   the input (`f(frame, frame)` is fine).
 - Every function returns 0 on success and -1 when the input is empty or has
   the wrong type; `dst` is untouched in that case.
 - Filters with a window (blur, Sobel, local max) leave a border of
   `radius` pixels: blurs copy the source there, Sobel writes zeros.
*/
#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>

// Alternative greyscale (CV_8UC1): 255 - round((B + G + R) / 3).
int myRGB2Gray(const cv::Mat& src, cv::Mat& dst);

// Sepia tone with a vignette that darkens towards the image edges.
int sepia(const cv::Mat& src, cv::Mat& dst);

// 5x5 Gaussian blur [1 2 4 2 1]^T [1 2 4 2 1] / 100, straightforward 2-D version.
int blur5x5_1(const cv::Mat& src, cv::Mat& dst);
// Same result as blur5x5_1 (exactly, away from the border) using two 1-D passes.
int blur5x5_2(const cv::Mat& src, cv::Mat& dst);

// 3x3 Sobel as two separable passes, CV_16SC3 output.
//  X: [-1 0 1] (columns) x [1 2 1]^T   positive where intensity increases to the right.
//  Y: [1 2 1] (columns) x [1 0 -1]^T   positive where intensity increases upwards
//     (the course convention; cv::Sobel's dy has the opposite sign).
int sobelX3x3(const cv::Mat& src, cv::Mat& dst);
int sobelY3x3(const cv::Mat& src, cv::Mat& dst);

// Gradient magnitude sqrt(sx^2 + sy^2) of two CV_16SC3 Sobel images, saturated to CV_8UC3.
int magnitude(const cv::Mat& sx, const cv::Mat& sy, cv::Mat& dst);

// Blur, then quantise every channel into `levels` levels (bucket b = 255 / levels).
int blurQuantize(const cv::Mat& src, cv::Mat& dst, int levels = 10);

// Zoom: crop a window twice the size of `face` around it and resize to 640x480 (in place on `tar`).
// Does nothing when the face already fills half the frame.
int centralize(cv::Mat& tar, cv::Rect& face);

// Simulates not being able to tell red from green: B' = (G + R) / 2, G' = R' = B.
int confuseRedGreen(const cv::Mat& src, cv::Mat& dst);

// 11x11 maximum filter (works for any 8-bit multi-channel image); 5-pixel border copied from src.
int localMax(const cv::Mat& src, cv::Mat& dst);

// dst = alpha * src + beta, saturated (contrast / brightness).
int brightAndContrast(const cv::Mat& src, cv::Mat& dst, float alpha, int beta);

// Kaleidoscope: the square image is cut into num x num tiles of `size` pixels
// and every tile is replaced by a 4-fold mirror image of its top-left quadrant
// (mask * tile * mask^T, with mask from makeKaleidoscopeMask).
cv::Mat makeKaleidoscopeMask(int size);
int kaleidoscope(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask, int num, int size);

#endif
