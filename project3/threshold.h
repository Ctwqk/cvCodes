#ifndef THRESHOLD_H
#define THRESHOLD_H 

#include <iostream>
#include <opencv2/opencv.hpp>

//given a color image, calculate the binary image
//src: origin color image
//dst: output binary image
//idx: which channel is used for calculate the grayscale image:
//0: Blue 1: Green 1: Red 4: opencv grayscale 
int toBinary(cv::Mat &src,cv::Mat &dst,int idx=-1);

#endif
