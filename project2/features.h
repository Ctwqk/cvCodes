#ifndef FEATURE_H
#define FEATURE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
using namespace std;
using namespace cv;


//retrive the center 7x7 part of the Mat src
int central7x7(Mat &src, Mat &feature);

//convert a cv::Mat into std::vector<float>
vector<float> mat2vec(Mat &src);

//R&G channels are used to prodece the histogram
int histogram(Mat &src, vector<float> &feature);

//4 corners and center are the ROI I chosen to concatenate and generate the histogram
int multiHist(Mat &src, vector<float> &feature);

//reture a flatten histogram, which is produced by orientation and magnitude of sobel filtered src
int texture(Mat &src,vector<float> &feature);

//given the directory of images(imgsName), and the name of target image(tarName), return the DNN embedding(feature) of target image
int searchCsv(string imgsName, string tarName, vector<float> &feature);

//do DFT and then retrieve histogram based on several ROIs of the frequency domain spectrum
int fourierHist(Mat &src,vector<float> &feature);  

//calculate the gabor filtered src and then calculate histogram based on that; gabor filter used the function provided by openCv
int gaborHist(Mat &src,vector<float> &feature);
		
#endif
