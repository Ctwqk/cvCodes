#include "preprocess.h"
using namespace std;
using namespace cv;

//cut off the border regions with BFS
//array: the flatted image
//row&col: the size of origin image
int cutEdgeRegion(int *array,int row,int col){
	//EDGE: the range that is regarded as border, any region that connect with border will be deleted.
	//size: the size of the Queue for BFS
	//endPoint: how many point you want to cut off at most
	int EDGE=5,size=4*(row+col),endPoint=row*col/6;
	endPoint=row*col;//I want to erase as much as possible
	int queue[size];
	int front=0,rear=0;
	int idx,idx2;
	//push the border into queue
	for(int i=0;i<=EDGE;i++){
		idx=(row-1-i)*col;
		idx2=i*col;
		for(int j=0;j<col;j++){
			if(i==EDGE){
				if(array[idx]>=0) queue[rear++]=idx;
				if(array[idx2]>=0) queue[rear++]=idx2;
			}
			array[idx++]=-2;
			array[idx2++]=-2;
		}
		for(int j=EDGE+1;j<row-EDGE-1;j++){
			idx=col*j;
			if(i==EDGE){
				if(array[idx+i]>=0) queue[rear++]=idx+i;
				if(array[idx+col-i-1]>=0) queue[rear++]=idx+col-i-1;
			}
			array[idx+i]=-2;
			array[idx+col-i-1]=-2;
		}
	}
	//4-connected
	int dir[][2]={{0,1},{1,0},{0,-1},{-1,0}/*,{1,-1},{-1,1},{1,1},{-1,-1}*/};

	int z,count=0;
	//BFS
	while(rear!=front&&count<endPoint){
		int tmp=queue[front%size];
		front++;
		idx=tmp/col;
		idx2=tmp%col;
		for(int i=0;i<4;i++){
			z=(idx+dir[i][1])*col+idx2+dir[i][0];
			if(array[z]>=0){
				array[z]=-2;
				count++;
				queue[rear%size]=z;
				rear++;
				if(rear-front>=size){
					cout<<"out of size"<<endl;
					return -1;
				}
			}
		}
		rear%=size;
		front%=size;
		if(front>rear) rear+=size;
	}
	//showArr(array,row,col);
	return 0;
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
	//dst=Mat(row,col,CV_8U,Scalar((uchar)upper));
	dst=Mat(row,col,CV_8U,Scalar(0));
	for(int i=0;i<row;i++){
		uchar* sPtr=src.ptr<uchar>(i);
	    uchar* dPtr=dst.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			//if((int)sPtr[j]<threshold) dPtr[j]=0;	
			if((int)sPtr[j]<threshold) dPtr[j]=upper;	
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
	if(idx==-1)idx=distance(errs,max_element(begin(errs),end(errs)));	
	//cout<<idx<<endl;
	//threshold(grays[idx],dst,pos[idx],255,THRESH_BINARY);
	int col=src.cols,row=src.rows;
	int *array=new int[col*row];
	uchar* ptr=grays[idx].ptr<uchar>(0);
	myThreshold(grays[idx],dst,pos[idx],255);
	for(int i=0;i<row*col;i++){
		array[i]=ptr[i];
	}
	cutEdgeRegion(array,row,col);
	for(int i=0;i<row*col;i++){
		ptr[i]=array[i];
	}
	//cout<<pos[idx]<<endl;
	return 0;
}
