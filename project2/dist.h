#ifndef DIST_H
#define DIST_H
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

float dist(vector<float> &feature1,const vector<float> &feature2);

float histDist(vector<float> &feature1,const vector<float> &feature2);

float cosDist(vector<float> &feature1, const vector<float> &feature2);

float myDist(vector<float> &feature1, const vector<float> &feature2);



#endif
