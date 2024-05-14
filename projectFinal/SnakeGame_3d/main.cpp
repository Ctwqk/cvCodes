/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * build patterns(descrtitor and keypoints) from pattern images
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include "saveOrReadPattern.hpp"

using namespace std;
using namespace cv;
int main(int argc,char** argv){
	if(argc<2){
		establishPatterns("patternimages","pattern.txt",400,1);
	}
	else{
		establishPatterns(argv[0],argv[1],400,1);
	}

	return 0;

}
