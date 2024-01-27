#ifndef FILTER_H
#define FILTER_H
#include "opencv2/opencv.hpp"
#include "FaceDetect/faceDetect.h"
#include <iostream>
#include <fstream>
#include <math.h>
int myRGB2Gray(const cv::Mat &src,cv::Mat &ans);
int sepia(cv::Mat &src,cv::Mat &ans);
int blur5x5_1(cv::Mat &src,cv::Mat &dst);
int blur5x5_2(cv::Mat &src,cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &src1,cv::Mat &src2,cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst,int levels=9);
int centralize(cv::Mat &src,cv::Rect &face);
int confuseRedGreen(cv::Mat &src,cv::Mat &dst);
int kaleidoscope(cv::Mat& src,cv::Mat &dst, cv::Mat &mask,int& num,int& size);
int localMax(cv::Mat& scr, cv::Mat& dst);
int brightAndContrast(cv::Mat &src,cv::Mat &dst,float& alpha, int &beta);
#endif
