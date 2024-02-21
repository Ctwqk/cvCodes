#include "features.h"
using namespace std;
using namespace cv;

float centralMoment_pq(Mat &map,RotatedRect &ans){
	int N=0,row=map.rows,col=map.cols;
	float xSum=0,ySum=0,xMean,yMean;
	int x[row*col],y[row*col]; 
	for(int i=0;i<row;i++){
		uchar* ptr=map.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			if(ptr[j]){
				xSum+=j;
				ySum+=i;
				x[N]=j;
				y[N]=i;
				N++;
			}
		}
	}
	xMean=xSum/N;
	yMean=ySum/N;
	auto u=[N,xMean,yMean](const int p,const int q,const int* x,const int* y){
		int sum=0;
		for(int i=0;i<N;i++){
			sum+=pow((x[i]-xMean),p)*pow((y[i]-yMean),q);
		}	
		return sum;
	};
	float theta=0.5*atan2(2*u(1,1,x,y),u(0,2,x,y)-u(2,0,x,y));
	float vx=cos(theta),vy=sin(theta);
	float maxX=0,maxY=0,minX=10000,minY=10000;
	float tmpX,tmpY;
	for(int i=0;i<N;i++){
		x[i]-=xMean;
		y[i]-=yMean;
		tmpX=vx*x[i]-vy*y[i];
		tmpY=vy*x[i]+vx*y[i];
		maxX=max(maxX,tmpX);
		maxY=max(maxY,tmpY);
		minX=min(minX,tmpX);
		minY=min(minY,tmpY);
	}
	tmpX=(maxX+minX)/2;
	tmpY=(maxY+minY)/2;
	xMean+=vx*tmpX+vy*tmpY;
	yMean+=vx*tmpY-vy*tmpX;
	ans=RotatedRect(Point2f(xMean,yMean),Size2f(maxX-minX,maxY-minY),-theta*180.0/M_PI);
	return theta;
}

int drawBox(Mat &src,RotatedRect &box,Mat &dst){
	Point2f vertices[4];
	box.points(vertices);
	for(int i=0;i<4;i++){
		line(dst,vertices[i],vertices[(i+1)%4],Scalar(255));
	}
}

