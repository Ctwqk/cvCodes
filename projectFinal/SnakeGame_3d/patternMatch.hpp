/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * given the patterns from a dataset, match current frame with one of them.
 */
#ifndef PATTERHMATCH_H
#define PATTERHMATCH_H
#include <iostream>
#include <opencv2/opencv.hpp>

int matchPattern(std::vector<cv::Mat> &patternDescriptors, cv::Mat &queryDescriptor,std::vector<std::vector<cv::KeyPoint>> &patternKeypoints, std::vector<cv::KeyPoint> &keypoints,cv::BFMatcher &matcher,std::vector<cv::DMatch> &matches,int &tar,cv::Mat &homography_matrix,int idx=0);

int getHomoMat(std::vector<cv::KeyPoint> &patternKeypoints,std::vector<cv::KeyPoint> &queryKeypoints,std::vector<cv::DMatch> &matches,cv::Mat &homography_matrix);

int getHomoMat(std::vector<cv::Point2f> &patternpoints,std::vector<cv::Point2f> &querypoints,cv::Mat &homography_matrix);



#endif
