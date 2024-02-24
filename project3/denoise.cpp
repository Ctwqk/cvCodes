#include "denoise.h"
using namespace std;
using namespace cv;
//operation: 0-grow 1-shrink
//kernelType: 1-all 0-interval
int cleanUp(Mat &src,Mat &dst,int kernelSize,int operationType, int kernelType){
	if(!(kernelSize%2)){
		cout<<"expecting odd kernel size"<<endl;
		return -1;
	}
	Mat kernel=Mat::ones(Size(kernelSize,kernelSize),CV_8U);
	dst=src.clone();
	if(!kernelType){
		for(int i=0;i<kernelSize;i++){
			uchar* ptr=kernel.ptr<uchar>(i);
			for(int j=0;j<kernelSize;j++){
				if(((i%2)^(j%2))){
					ptr[j]=0;
				}
			}
		}
	}
	kernel.at<uchar>(kernelSize/2,kernelSize/2)=0;
	
	//for debu;
	src/=255;
	int row=src.rows,col=src.cols,threshold=(float)kernelSize*kernelSize/4/(1+kernelType);
	//threshold=1;
	Mat tmp=Mat::ones(kernel.size(),CV_8U);
	Mat res;
	tmp*=operationType;
	for(int i=0;i<=row-kernelSize;i++){
		for(int j=0;j<=col-kernelSize;j++){
			if(src.at<uchar>(i+kernelSize/2,j+kernelSize/2)!=operationType)continue;
			bitwise_xor(tmp,src(Rect(j,i,kernelSize,kernelSize)),res);
			bitwise_and(res,kernel,res);
			if(sum(res)[0]>threshold){
				dst.at<uchar>(i+kernelSize/2,j+kernelSize/2)=1-operationType;
			}
		}
	}
	dst*=255;
	return 0;
}
	
	

int denoise(Mat &src, Mat &dst){
	Mat tmp1,tmp2;
	if(cleanUp(src,tmp1,3,1,0)) cout<<"cleanUp reports error"<<endl;
	if(cleanUp(tmp1,tmp2,3,0,1)) cout<<"cleanUp reports error"<<endl;
	if(cleanUp(tmp2,tmp1,5,1,0)) cout<<"cleanUp reports error"<<endl;
	if(cleanUp(tmp1,tmp2,5,0,1)) cout<<"cleanUp reports error"<<endl;
	if(cleanUp(tmp2,tmp1,3,1,0)) cout<<"cleanUp reports error"<<endl;
	if(cleanUp(tmp1,dst,3,0,1)) cout<<"cleanUp reports error"<<endl;
	return 0;
}

