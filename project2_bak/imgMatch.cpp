#include <iostream>
#include <opencv2/opencv.hpp>
#include "dist.h"
#include "csv_util.h"
#include "features.h"
using namespace std;
using namespace cv;


int main(int argc,char **argv){
	Mat	img=imread(argv[1]);

	Mat feature;
	central7x7(img,feature);
	
	vector<vector<float>> features;
	vector<string> fileNames;
	read_image_data_csv("imgCsv", 	fileNames, features);
	vector<pair<float, int>> rank;
	
	for(int i=0;i<fileNames.size();i++){
		float tmp=dist(features[i],feature);
		rank.push_back({tmp,i});
	}
	sort(rank.begin(),rank.end(),greater<float>());
	fig1=imread(rank[0],IMREAD_COLOR);
	fig2=imread(rank[0],IMREAD_COLOR);
	fig3=imread(rank[0],IMREAD_COLOR);
	imshow("fig1", fig1);
	imshow("fig2", fig2);
	imshow("fig3", fig3);
	waitKey(0);
	return 0;

	
}
	
