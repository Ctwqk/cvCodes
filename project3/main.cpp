#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "cleanup.h"
#include "region.h"
using namespace std;
using namespace cv;

int main(int argc, char** argv){
	Mat dst,src=imread(argv[1]);
//	resize(src,src,Size(50,50),0,0);
	/*
	Mat planes[3];
	split(src,planes);
	for(int i=0;i<3;i++){
		imshow(to_string(i),planes[i]);
	}
	//cvtColor(src,src,COLOR_BGR2GRAY);
	Mat planes[4];
	split(src,planes);
	cvtColor(src,planes[3],COLOR_BGR2GRAY);
	for(int i=0;i<4;i++) imshow(to_string(i),planes[i]);
	*/
	if(argc==2)	toBinary(src,dst);
	if(argc==3) toBinary(src,dst,atoi(argv[2]));
	//adaptiveThreshold(planes[3],dst,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,71,6);

	imshow("src",src);
	for(int i=0;i<dst.rows;i++){
		uchar* ptr=dst.ptr<uchar>(i);
		for(int j=0;j<dst.cols;j++){
			ptr[j]?ptr[j]=0:ptr[j]=255;
		}
	}
	imshow("bin",dst);
	denoise(dst,dst);
	imshow("pur",dst);
	
	regionize(dst,dst,2);
	
	imshow("reg",dst);
	mapToColor(dst,dst,3);
	imshow("colorReg",dst);

	waitKey(0);
	return 0;
}
