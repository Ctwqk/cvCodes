#ifndef FEATURE_H
#define FEATURE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
using namespace std;
using namespace cv;
int central7x7(Mat &src, Mat &feature);
vector<Mat> getFeatures(Mat &src);



#endif
