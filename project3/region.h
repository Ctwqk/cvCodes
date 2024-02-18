#ifndef REGION_H
#define REGION_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <set>
int regionize(cv::Mat &src,cv::Mat &dst,int num=0);
int mapToColor(cv::Mat &map,cv::Mat &dst,int num=3);




#endif
