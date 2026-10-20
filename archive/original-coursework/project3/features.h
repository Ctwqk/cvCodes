/*
 * Author: Taiwei Cui
 * Time: Feb 27 2024
 *
 * */
#ifndef FEATURES_H
#define FEATURES_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "csv_util.h"
#include "NetWorkThings/dnnEmbedding.h"

//this function will calculate and return the feature vector of given image map //map: given image map
//ans: hold the oriented bounding box of the region
//savePath: the CSV file in which saves the feature vector ( if is given)
//label: the label to which the region belong ( if is given)
std::vector<float> featureBox(cv::Mat &map, cv::RotatedRect &ans,std::string savePath="",std::string label="");

//this function draw the orientated box and the axis with minimal moment
//box: the oriented bounding box
//features: the features to be show
//dst: the image that bounding box,axis and text will be painted on
//name: the label of this region
int drawBox(cv::RotatedRect &,std::vector<float>, cv::Mat &,std::string name="");


#endif
