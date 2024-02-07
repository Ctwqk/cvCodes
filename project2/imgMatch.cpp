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
	vector<char * > fileNames;
	cout<<"0";
	read_image_data_csv("imgCsv", 	fileNames, features);
	cout<<"1";
	vector<pair<float, int>> rank;
	cout<<"2";
	
	for(int i=0;i<fileNames.size();i++){
		float tmp=dist(features[i],feature);
		rank.push_back({tmp,i});
	}
	sort(rank.begin(),rank.end(),[](const pair<float,int> &a,const pair<float,int> &b){return a.first>b.first;});
	Mat fig1=imread(fileNames[rank[0].second],IMREAD_COLOR);
	Mat fig2=imread(fileNames[rank[1].second],IMREAD_COLOR);
	Mat fig3=imread(fileNames[rank[2].second],IMREAD_COLOR);
	imshow("fig1", fig1);
	imshow("fig2", fig2);
	imshow("fig3", fig3);
	waitKey(0);
	return 0;
}
	
