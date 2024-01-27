#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
using namespace std;
using namespace cv;
int main(){
	cv::Mat pic=imread("cathedral.jpeg",IMREAD_COLOR);
	pic=pic(cv::Rect(0,0,400,400));
	cv::Mat tri=Mat::eye(400,400,CV_32F);
	cout<<pic.type()<<endl;

	Mat ans;
	vector<Mat> channels;
	split(pic,channels);
	flip(tri,tri,0);
	tri=tri+Mat::eye(400,400,CV_32F);
	imshow("before",pic);
	channels[1]+=100;
	channels[0]+=100;
	channels[2]+=100;
	Mat tmp;
	merge(channels,tmp);	
	
	imshow("wtf",tmp);
	cout<<tri.size()<<endl;
	cout<<tri.type()<<" "<<channels[0].type()<<endl;
	for(int i=0;i<3;i++){
		
		channels[i].convertTo(channels[i],CV_32F);
		for(int x=0;x<200;x++){
			float* cPtr=channels[i].ptr<float>(x);
			float* cPtr1=channels[i].ptr<float>(x+200);
			for(int y=0;y<200;y++){
				cPtr[y+200]=0;
				cPtr1[y]=0;
				cPtr1[y+200]=0;
			}
		}



		gemm(channels[i],tri,1.0,cv::Mat(),0.0,channels[i]);
		gemm(tri,channels[i],1.0,cv::Mat(),0.0,channels[i]);
		channels[i].convertTo(channels[i],CV_8U);
		cout<<i<<endl;
	}
	merge(channels,pic);
	imshow("newWIn",pic);






	waitKey(0);
	return 0;
}
