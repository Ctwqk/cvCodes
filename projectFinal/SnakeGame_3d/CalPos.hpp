/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 */
#ifndef CALPOS_H
#define CALPOS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>

int CalPos(cv::Mat &frame,cv::Mat &tvec,cv::Mat &rvec,cv::Mat &camera_matrix,std::vector<double> &distCoef,int x,int y);



#endif
