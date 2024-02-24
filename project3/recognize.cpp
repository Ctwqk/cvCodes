#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "dist.h"
#include "features.h"
#include "threshold.h"
#include "region.h"
#include "denoise.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	if(argc<2){
		cout<<"expacting input"<<endl;
		return -1;
	}
	int mode=0;
	if(argc==3) mode=atoi(argv[2]);
	

	string filePath="trainset/featureCsv";
	char* filePath_ =new char[filePath.size()+1];
	strcpy(filePath_,filePath.c_str());
	vector<float> feature;
	Mat src=imread(argv[1]);
	toBinary(src,src);
	denoise(src,src);
	vector<Mat> maps=regionize(src,src,1);
	RotatedRect tmp;
	if(mode){
		char label[128];
		cout<<"please input the label:"<<endl;
		cin>>label;	
		featureBox(maps[0],tmp,filePath_,label);
	}
	feature=featureBox(maps[0],tmp);
	vector<vector<float>> features;
	vector<char*> labels;
	read_image_data_csv(filePath_,labels,features);
	string label=string(labels[distance(features.begin(),max_element(features.begin(),features.end(),[&feature](const vector<float> &a,const vector<float> &b){return dist(feature,a)>dist(feature,b);}))]);
	cout<<label<<endl;
}
