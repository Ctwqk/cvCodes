#ifndef DNNEMBEDDING_H
#define DNNEMBEDDING_H
#include <cstdio>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "opencv2/opencv.hpp"  // the top include file
#include "opencv2/dnn.hpp"


int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug );


#endif
