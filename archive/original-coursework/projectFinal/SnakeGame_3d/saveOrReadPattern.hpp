/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * save to or read from given .txt file the patterns of images.
 */
#ifndef SAVEORREADPATTERN_HPP
#define SAVEORREADPATTERN_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <opencv2/xfeatures2d.hpp>


int writePatternToCsv(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor,int &height,int&width, std::string filepath="pattern.txt",int mode=0);

int readPatternsFromCsv(std::string csvFile,std::vector<std::vector<cv::KeyPoint>> &keypoints, std::vector<cv::Mat>&descriptors,std::vector<cv::Mat> &mat_3d, std::vector<std::vector<cv::Point3f>> &points3d);

int establishPatterns(std::string imageFilePath, std::string patternFilePath="pattern.txt",int minHessian = 400,int mode=0);

#endif
