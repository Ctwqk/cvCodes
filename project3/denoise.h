#ifndef DENOISE_H
#define DENOISE_H
#include <iostream>
#include <opencv2/opencv.hpp>

//by using kernels, this function will clean the image up
//src: origin image
//dst: cleaned-up image
int denoise(cv::Mat &, cv::Mat &);







#endif
