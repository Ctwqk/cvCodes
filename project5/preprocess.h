#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include <opencv2/opencv.hpp>
//int cutEdgeRegion(int *array,int row,int col);
int toBinary(cv::Mat &src,cv::Mat &dst,int idx=-1);

#endif
