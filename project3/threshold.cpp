#include "threshold.h"

using namespace std;
using namespace cv;

//using kmean to seperate the pixels to 2 and use the medium of 2 centroid as threshold
//tmp: input image
//pos: the threshold
float kmeanThreshold(Mat &tmp,int &pos){
	int row=tmp.rows,col=tmp.cols;
	float TOL=0.00001;
	float centroids[]={0,255};	
	int nums[]={0,0};
	int bins[]={0,0};
	float err=100,tmpX,tmpY;
	float center;
	while(err>TOL){
		nums[0]=0;
		nums[1]=0;
		bins[0]=0;
		bins[1]=0;
		center=(centroids[0]+centroids[1])/2;
		for(int i=0;i<row;i++){
			const uchar* ptr=tmp.ptr<uchar>(i);
			for(int j=0;j<col;j++){
				if(ptr[j]>center){
					nums[1]++;
					bins[1]+=ptr[j];
				}
				else {
					nums[0]++;
					bins[0]+=ptr[j];
				}
			}
		}
		tmpX=centroids[0]-(float)bins[0]/nums[0];
		tmpY=centroids[1]-(float)bins[1]/nums[1];	
		err=tmpX>tmpY?tmpY:tmpX;
		centroids[0]+=tmpX;
		centroids[1]+=tmpY;
	}
	err=(centroids[0]+centroids[1])/2;
	pos=err;
	return err;	
}

//directly using the average value of image as threshold
//tmp: origin image
//pos: threshold
float averageThreshold(Mat &tmp,int &pos){
	int row=tmp.rows,col=tmp.cols;
	float threshold,sum;
	for(int i=0;i<row;i++){
		uchar* ptr=tmp.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			sum+=ptr[j];
		}
	}
	threshold=sum/(row*col);	
	pos=threshold;
	return threshold;
}

//use otsu method. Find the threshold that miximize the inter-class variance
//tmp: origin image
//pos: threshold
float otsuThreshold(Mat &tmp,int &pos){
	float hist[256]={},col=tmp.cols,row=tmp.rows,theta[256]={};
	float sum=0,scale=col*row, sum1=0;
	for(int i=0;i<row;i++){
		uchar* ptr=tmp.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			hist[ptr[j]]++;
		}
	}
	for(int i=0;i<256;i++){
		hist[i]/=scale;
		sum1+=i*hist[i];
		sum+=hist[i];
	}
	float wl=0,wr=sum,mud,tmpl=0,tmpr=sum1;
	float max=0,icv;
	int idx=0;
	//using dynamic programing to accelerate the progress
	for(int i=0;i<255;i++){
		wl+=hist[i];
		wr-=hist[i];
		tmpl+=i*hist[i];
		tmpr-=i*hist[i];
		mud=tmpl/wl-tmpr/wr;
		icv=mud*mud*wl*wr;
		if(icv>max){
			max=icv;
			idx=i;
		}
	}
	pos=idx;
	return max;
}

//given an image and a threshold, depart the image into foreground and background
//src: origin image
//dst: output binary image
//threshold: thresohld:
//upper: the value for foreground pixels
int myThreshold(Mat &src,Mat &dst, int threshold, int upper){
	if(!threshold) threshold++;
	int row=src.rows,col=src.cols;
	dst=Mat(row,col,CV_8U,Scalar((uchar)upper));
	for(int i=0;i<row;i++){
		uchar* sPtr=src.ptr<uchar>(i);
	    uchar* dPtr=dst.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			if((int)sPtr[j]<threshold) dPtr[j]=0;	
		}
	}
	return 0;
}

//given a color image, calculate the binary image
//src: origin color image
//dst: output binary image
//idx: which channel is used for calculate the grayscale image:
//0: Blue 1: Green 1: Red 4: opencv grayscale 
int toBinary(Mat &src,Mat &dst,int idx){
	Mat tmp=src.clone();
	Mat grays[4];
	split(src,grays);
	cvtColor(src,grays[3],COLOR_BGR2GRAY);
	//  float err=kmeanThreshold(tmp);	
	float errs[4]={};
	int pos[4]={};
	//choose the channel that gives the largest interclass variance
	for(int i=0;i<4;i++){
		errs[i]=otsuThreshold(grays[i],pos[i]);
		//cout<<errs[i]<<endl;
	}
	//in pratice I found the interclass variance from grayscale channel should be doubled
	errs[3]*=2;
	if(idx<0)idx=distance(errs,max_element(begin(errs),end(errs)));	
	//cout<<idx<<endl;
	//threshold(grays[idx],dst,pos[idx],255,THRESH_BINARY);
	myThreshold(grays[idx],dst,pos[idx],255);
	//cout<<pos[idx]<<endl;
	return 0;
}
