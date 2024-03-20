/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 * this file use opencv functions to calculate harris corners as robust feature and find robust features based on given threshold
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
int main(int argc,char** argv){
	VideoCapture *cmrCap=new VideoCapture(atoi(argv[1]));
	if(!cmrCap->isOpened()){
		cout<<"failed to open the image"<<endl;
		return -1;
	}
	Mat frame,src;
	int THRESHOLD=160;// user-setting threshold

	for(;;){
		char c=waitKey(10);
		if(c=='q'){
			destroyAllWindows();
			waitKey(1);
			break;
		}
		*cmrCap>>frame;
		cvtColor(frame,src,COLOR_BGR2GRAY);
		cornerHarris(src,src,3,3,0.05,BORDER_DEFAULT);//calculating harris corner map
		normalize(src,src,0,255,NORM_MINMAX,CV_32F,cv::Mat());
		for(int i=0;i<src.rows;i++){// selecting those points with a value larger than the threshold
			float *ptr=src.ptr<float>(i);
			for(int j=0;j<src.cols;j++){
				if(ptr[j]>THRESHOLD) circle(frame,Point(j,i),5,Scalar(0,0,255));
			}
		}
		imshow("vid",frame);

	}
	return 0;
}

