#include <iostream>
#include <opencv2/opencv.hpp>
#include "cnpy.h"
#include <vector>
using namespace std;
using namespace cv;

int main(int argc,char** argv){
	cnpy::NpyArray filters=cnpy::npy_load("filters.npy");
	cnpy::NpyArray image=cnpy::npy_load("image.npy");
	Mat img(image.shape[0],image.shape[1],CV_32F,image.data<float>());
	vector<Mat> filter2ds;
	int row=filters.shape[1];
	int col=filters.shape[2];
	int layer=filters.shape[0];
	int num=col*row;
	vector<float> ans;
	Mat tmp;

	for(int i=0;i<layer;i++){
		filter2ds.push_back(Mat(row,col,CV_32F,filters.data<float>()+i*num));
		filter2D(img,tmp,-1,filter2ds.back());
		ans=vector<float>(tmp.begin<float>(),tmp.end<float>());
		cnpy::npy_save("ansImage"+to_string(i+1)+".npy",ans.data(),{image.shape[0],image.shape[1]},"w");

	}
	return 0;
}
