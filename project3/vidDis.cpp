/*
 *  Author: taiwei cui
 *  Time: Feb 27 2024
 *  
 * */

/*
 * this shows the video captured by a webcam and 
 *	when system running, you can use 1-6 to determine how many region you want to show at most. OR you can use 1-6 to choose which region to focus on so that you can label this region. 
 *	to toggle the function of key 1-6 (setting number of region or which region) , press 't'.
 *	to quite the video press 'q'
 *	to label current focusing region, press 's'
 *
 * */


#include <iostream>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "denoise.h"
#include "region.h"
#include "features.h"
#include "csv_util.h"
#include "dist.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	int mode=0;
	if(argc>1) mode=atoi(argv[1]);
	VideoCapture *capdev;
	capdev=new VideoCapture(2);
	if(!capdev->isOpened()){
		cout<<"failed to open the camera"<<endl;
		return -1;
	}
	Mat frame,src;
	int idx;
	string filePath="trainset/featureCsv";
	
	char* filePath_=new char[filePath.size()+1];
	strcpy(filePath_,filePath.c_str());

	vector<vector<float>> features;
	vector<char*> labels;
	read_image_data_csv(filePath_,labels, features);

	char labelName[128];
	int flag;
	int selectNum=6;
	int targetNum=6;
	int toggle=1; //-1: number key will indicate which region to focus on
				  //1: number key will indicate how many region(maximum) you want to show
	Mat target;

	RotatedRect tmp;
	vector<float> curFeature;
	string model[2]={"num","tar"};
	auto comp=[&curFeature](const vector<float> &a,const vector<float> &b){
		return cosDist(curFeature,a)>cosDist(curFeature,b);
	};
	string name="";
	for(;;){
		flag=0;
		name="";
		*capdev>>src;
		char key=waitKey(10);

		
		if(key=='q') break;
		else if(toggle==-1&&key=='s'){
			cout<<"please input the label of current frame"<<endl;
			cin>>labelName;
			flag=1;	
		}
		else if(key-'0'>0&&key-'0'<=6){
			if(toggle==1){
				selectNum=key-'0';
				cout<<"num"<<endl;
			}
			else {
				targetNum=key-'0'-1;
				cout<<"tar"<<endl;
			}
		}
		else if(key=='t'){
			toggle*=-1;
		}
		idx=toBinary(src,frame);
		imshow("bin",frame);
		denoise(frame,frame);
		imshow("denoised",frame);
		vector<Mat> maps=regionize(frame,frame,selectNum);
		imshow("reg",frame);
		if(toggle==-1){
			if(targetNum>=maps.size()) targetNum=maps.size()-1;
			target=maps[targetNum];
			//if flag!=0, the feature will be saved into the CSV
			curFeature=flag?featureBox(target,tmp,filePath_,labelName):featureBox(target,tmp);
			if(mode){
				//nearest neighbor
				name=string(labels[distance(features.begin(),max_element(features.begin(),features.end(),comp))]);
			}
			//drawing bunding box and axis
			drawBox(tmp,curFeature,frame,name);
			drawBox(tmp,curFeature,src,name);
		}
		else{
			for(Mat m:maps){
				//if flag!=0, the feature will be saved into the CSV
				curFeature=flag?featureBox(m,tmp,filePath_,labelName):featureBox(m,tmp);
				if(mode){
					//nearest neighbor
					name=string(labels[distance(features.begin(),max_element(features.begin(),features.end(),comp))]);
				}
				//drawing bunding box and axis
				drawBox(tmp,curFeature,frame,name);
				drawBox(tmp,curFeature,src,name);
			}
		}
		imshow("src",src);
		imshow("video",frame);
	}

	delete capdev;
	delete[] filePath_;
	return 0;
}
