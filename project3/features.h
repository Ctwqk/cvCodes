#ifndef FEATURES_H
#define FEATURES_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "csv_util.h"
#include "NetWorkThings/dnnEmbedding.h"

std::vector<float> featureBox(cv::Mat &map, cv::RotatedRect &ans,std::string savePath="",std::string label="");

int drawBox(cv::RotatedRect &,std::vector<float>, cv::Mat &);






#endif
