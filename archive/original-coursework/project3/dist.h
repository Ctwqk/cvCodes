#ifndef DIST_H
#define DIST_H
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

//return the Euclidean distance of two given vector, require same sie
float dist(const vector<float> &feature1,const vector<float> &feature2);

//return the histogram intersection of two histogram,require same size
float histDist(vector<float> &feature1,const vector<float> &feature2);

//return the cosDist of two given vector, require same size
float cosDist(const vector<float> &feature1, const vector<float> &feature2);

//directly usage of cosDict
float myDist(vector<float> &feature1, const vector<float> &feature2);



#endif
