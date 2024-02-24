#include <iostream>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "denoise.h"
#include "region.h"
#include "features.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	VideoCapture *capdev;
	capdev=new VideoCapture(0);
	if(!capdev->isOpened()){
		cout<<"failed to open the camera"<<endl;
		return -1;
	}
	Mat frame;
	int idx;
	string filePath="trainset/featureCsv";
	char* filePath_=new char[filePath.size()+1];
	strcpy(filePath_,filePath.c_str());
	char labelName[128];
	int flag;
	int selectNum=6;
	int targetNum=6;
	int toggle=1;
	Mat target;
	RotatedRect tmp;
	vector<float> curFeature;
	string model[2]={"num","tar"};
	for(;;){
		flag=0;
		*capdev>>frame;
		char key=waitKey(10);
		

		if(key=='q') break;
		else if(key=='s'){
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
		idx=toBinary(frame,frame);
		denoise(frame,frame);
		vector<Mat> maps=regionize(frame,frame,selectNum);
		if(toggle==-1){
			if(targetNum>=maps.size()) targetNum=maps.size()-1;
			target=maps[targetNum];
			curFeature=flag?featureBox(target,tmp,filePath_,labelName):featureBox(target,tmp);
			drawBox(tmp,curFeature,frame);
		}
		else{
			for(Mat m:maps){
				curFeature=flag?featureBox(m,tmp,filePath_,labelName):featureBox(m,tmp);
				drawBox(tmp,curFeature,frame);
			}
		}
		imshow("video",frame);
	}

	delete capdev;
	delete[] filePath_;
	return 0;
}
