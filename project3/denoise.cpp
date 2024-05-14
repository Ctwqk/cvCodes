#include "denoise.h"
using namespace std;
using namespace cv;
//this is actually a cleaning-up kernel 
//src: origin binary image
//dst: result binary image
//kernelSize: size of kernel
//operation: 0-grow 1-shrink
//kernelType: 1-all(4 connect if kernelSize is 3) 0-interval(8 connect if kernelSize is 3)
int cleanUp(Mat &src,Mat &dst,int kernelSize,int operationType, int kernelType){
	if(!(kernelSize%2)){
		cout<<"expecting odd kernel size"<<endl;
		return -1;
	}
	//build the kernel
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
	
	src/=255;
	int row=src.rows,col=src.cols,threshold=(float)kernelSize*kernelSize/4/(1+kernelType);
	//threshold=1;
	Mat tmp=Mat::ones(kernel.size(),CV_8U);
	Mat res;
	tmp*=operationType;
	for(int i=0;i<=row-kernelSize;i++){
		uchar* ptr=src.ptr<uchar>(i+kernelSize/2);
		uchar* ptr1=dst.ptr<uchar>(i+kernelSize/2);
		for(int j=0;j<=col-kernelSize;j++){
			if(ptr[j+kernelSize/2]!=operationType)continue;
			bitwise_xor(tmp,src(Rect(j,i,kernelSize,kernelSize)),res);
			bitwise_and(res,kernel,res);
			if(sum(res)[0]>threshold){
				ptr1[j+kernelSize/2]=1-operationType;
			}
		}
	}
	dst*=255;
	return 0;
}
	
	
//by using kernels, this function will clean the image up
//src: origin image
//dst: cleaned-up image
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

