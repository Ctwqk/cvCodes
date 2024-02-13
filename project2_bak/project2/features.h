#ifndef FEATURE_H
#define FEATURE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
using namespace std;
using namespace cv;
int central7x7(Mat &src, Mat &feature);
//vector<Mat> getFeatures(Mat &src);

vector<float> mat2vec(Mat &src);

int histogram(Mat &src, vector<float> &feature);

int multiHist(Mat &src, vector<float> &feature);

int texture(Mat &src,vector<float> &feature);
#endif

int searchCsv(string imgsName, string tarName, vector<float> &feature);
