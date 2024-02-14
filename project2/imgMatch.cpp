#include <iostream>
#include <opencv2/opencv.hpp>
#include "dist.h"
#include "csv_util.h"
#include "features.h"
#include <fstream>
using namespace std;
using namespace cv;


int main(int argc,char **argv){
	Mat	img;
	Mat feature;
	vector<float> vecFeature,vecFeature2;	
	vector<vector<float>> features;
	vector<char * > fileNames;
	string targetFeature;

	//this part fetch the target image, and the path to the csv file of given feature, and calculate the given type of feature of target image
	if(string(argv[3])=="7x7central") {
		img=imread(argv[1]);
		targetFeature=string(argv[2])+"/7x7csv";		
		central7x7(img,feature);
		vecFeature=mat2vec(feature);
	}
	else if(string(argv[3])=="histogram"){
		img=imread(argv[1]);
		targetFeature=string(argv[2])+"/histcsv";
		histogram(img,vecFeature);
	}
	else if(string(argv[3])=="multihist"){
		img=imread(argv[1]);
		targetFeature=string(argv[2])+"/multihistcsv";
		multiHist(img,vecFeature);
	}
	else if(string(argv[3])=="texturecolor"){
		img=imread(argv[1]);
		targetFeature=string(argv[2])+"/textcolor";
		texture(img,vecFeature);
		histogram(img,vecFeature2);
		vecFeature.insert(vecFeature.end(),vecFeature2.begin(),vecFeature2.end());
	}
	else if(string(argv[3])=="gabor"){
		img=imread(argv[1]);
		targetFeature=string(argv[2])+"/gaborhist";
		gaborHist(img,vecFeature);
	}
	else if(string(argv[3])=="dnn"){
		img=imread(string(argv[1]),IMREAD_COLOR);
		targetFeature=string(argv[2])+"/ResNet18_olym.csv";
	}
	else if(string(argv[3])=="mycustom"){
		img=imread(string(argv[1]),IMREAD_COLOR);
		targetFeature=string(argv[2])+"/mycustom";
		fourierHist(img,vecFeature);
		searchCsv(string(argv[2])+"/ResNet18_olym.csv",string(argv[1]),vecFeature2);
		//cout<<vecFeature.size()<<" "<<vecFeature2.size()<<endl;
		vecFeature.insert(vecFeature.end(),vecFeature2.begin(),vecFeature2.end());
	}
		
	//this part check whether the path of csv file exist, if not, return error information
	ifstream file(targetFeature);
	if(!file){
		cout<<"database not found"<<endl;
		return -1;
	}
	char * targetFeature_=new char[targetFeature.length()+1];
	strcpy(targetFeature_,targetFeature.c_str());
	read_image_data_csv(targetFeature_, 	fileNames, features);
	vector<pair<float, int>> rank;

	//if the path exist, based on the type of feature, calculate the distance between every images and target image respectively
	if(string(argv[3])=="7x7central"){
		for(int i=0;i<fileNames.size();i++){
			float tmp=dist(features[i],vecFeature);
			rank.push_back({tmp,i});
		}
	}
	else if(string(argv[3])=="texturecolor"||string(argv[3])=="multihist"||string(argv[3])=="histogram"||string(argv[3])=="gabor"){

		for(int i=0;i<fileNames.size();i++){
			float tmp=histDist(features[i],vecFeature);
			rank.push_back({tmp,i});
		}
	}
	
	else if(string(argv[3])=="dnn"){
		string tmp=string(argv[1]);
		for(int i=0;i<fileNames.size();i++){
			if(tmp.find(fileNames[i])!=string::npos){
				vecFeature=features[i];
				break;
			}
		}		
		for(int i=0;i<fileNames.size();i++){
			float tmp=dist(features[i],vecFeature);
			rank.push_back({tmp,i});
		}

	}
	else if(string(argv[3])=="mycustom"){
		//for(auto i : vecFeature) cout<<i<<" ";
		cout<<endl;
		for(int i=0;i<fileNames.size();i++){
			float tmp=myDist(features[i],vecFeature);
			rank.push_back({tmp,i});
		}
		
	}
	if(img.type()!=CV_8UC3) img.convertTo(img,CV_8UC3);
	resize(img,img,Size(600,480));
	imshow("src",img);
		
	//sort the result of distances, and show the given number of closest images
	sort(rank.begin(),rank.end(),[](const pair<float,int> &a,const pair<float,int> &b){return a.first>b.first;});
	vector<Mat> results(atoi(argv[4]));
	string targetName;
	for(int i=0;i<atoi(argv[4]);i++){
		if(string(argv[3])=="dnn"){
			targetName="images/"+string(fileNames[rank[i+1].second]);
		}
		else{
			targetName=string(fileNames[rank[i+1].second]);
		}
		results[i]=imread(targetName,IMREAD_COLOR);
		cout<<targetName<<endl;
		imshow("fig"+to_string(i),results[i]);
	}
	//show the furthest 3 images
	/*
	cout<<"press any key to show least possible 3 images"<<endl;
	waitKey(0);
	for(int i=0;i<3;i++){
		if(string(argv[3])=="dnn"){
			targetName="images/"+string(fileNames[rank[rank.size()-i-1].second]);
		}
		else{
			targetName=string(fileNames[rank[rank.size()-i-1].second]);
		}
		results[i]=imread(targetName,IMREAD_COLOR);
		cout<<targetName<<endl;
		imshow("fig -"+to_string(i),results[i]);
	}
	*/
	waitKey(0);

	return 0;
}
