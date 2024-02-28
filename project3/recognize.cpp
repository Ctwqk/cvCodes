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
	Mat src=imread(argv[1],1);
	toBinary(src,src);
	denoise(src,src);
	vector<Mat> maps=regionize(src,src,1);
	RotatedRect tmp;
	if(mode){
		char label[128];
		imshow("current image",src);
		cout<<"please input the label:"<<endl;
		cin>>label;	
		featureBox(maps[0],tmp,filePath_,label);
		waitKey(0);
		return 1;
	}
	feature=featureBox(maps[0],tmp);
	vector<vector<float>> features;
	vector<char*> labels;
	read_image_data_csv(filePath_,labels,features);

	auto distClassify=[&filePath_,&labels,&features](const vector<float>& feature){
		return string(labels[distance(features.begin(),max_element(features.begin(),features.end(),[&feature](const vector<float> &a,const vector<float> &b){return dist(feature,a)>dist(feature,b);}))]);
	};
	/*
	auto knnClassify=[&filePath_,&labels,&features](const vector<float>& feature,const int &k=5){
		vector<pair<float,string>> aux;
		for(int i=0;i<features.size();i++){
			aux.push_back({dist(features[i],feature),string(labels[i])});
			nth_element(aux.begin(),aux.begin()+k,aux.end(),greater<pair<float,char*>>());
		}
		map<string,int> check;
		int max=1;
		string target;
		for(int i=0;i<k;i++){
			if(check.find(aux[i].second)==check.end())check[aux[i].second]=1;
			else check[aux[i].second]++;
			if(check[aux[i].second]>max){
				max=check[aux[i].second];	
				target=aux[i].first;
			}
		}
		return target;
	};
	*/
	cout<<distClassify(feature);
	return 1;
}
