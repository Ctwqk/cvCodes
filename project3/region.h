#ifndef REGION_H
#define REGION_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <set>
#include <map>
std::vector<cv::Mat> regionize(cv::Mat &src,cv::Mat &dst,int num=6);
//int mapToColor(cv::Mat &map,cv::Mat &dst,int num=6);




#endif
