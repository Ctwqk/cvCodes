#include "threshold.h"

using namespace std;
using namespace cv;

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
		tmpX=(float)bins[0]/nums[0];
		tmpY=(float)bins[1]/nums[1];	
		err=min(abs(tmpX-centroids[0]),abs(tmpY-centroids[1]));
		centroids[0]=tmpX;
		centroids[1]=tmpY;
	}
	err=(centroids[0]+centroids[1])/2;
	pos=err;
	return err;	
}

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
	float wl=0,wr=sum,mul,mur,tmpl=0,tmpr=sum1;
	float max=0,icv;
	int idx=0;
	for(int i=0;i<255;i++){
		wl+=hist[i];
		wr-=hist[i];
		tmpl+=i*hist[i];
		tmpr-=i*hist[i];
		mul=tmpl/wl;
		mur=tmpr/wr;
		icv=(mul-mur)*(mul-mur)*wl*wr;
		if(icv>max){
			max=icv;
			idx=i;
		}
	}
	pos=idx;
	return max;
}
int myThreshold(Mat &src,Mat &dst, int threshold, int upper){
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

int toBinary(Mat &src,Mat &dst,int idx){
	Mat tmp=src.clone();
	Mat grays[4];
	split(src,grays);
	cvtColor(src,grays[3],COLOR_BGR2GRAY);
	//  float err=kmeanThreshold(tmp);	
	float errs[4]={};
	int pos[4]={};
	for(int i=0;i<4;i++){
		errs[i]=otsuThreshold(grays[i],pos[i]);
		//cout<<errs[i]<<endl;
	}
	errs[3]*=2;
	if(idx<0)idx=distance(errs,max_element(begin(errs),end(errs)));	
	//cout<<idx<<endl;
	//threshold(grays[idx],dst,pos[idx],255,THRESH_BINARY);
	myThreshold(grays[idx],dst,pos[idx],255);
	//cout<<pos[idx]<<endl;
	return 0;
}
