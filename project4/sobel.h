#ifndef SOBEL_H
#define SOBEL_H
#include <iostream>
#include <opencv2/opencv.hpp>

int integralImg(cv::Mat &src,cv::Mat &itgImg);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);	

#endif
