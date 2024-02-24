#include "features.h"
using namespace std;
using namespace cv;

vector<float> featureBox(Mat &map,RotatedRect &ans,string savePath,string label){
	int N=0,row=map.rows,col=map.cols;
	float xSum=0,ySum=0,xMean,yMean;
	int x[row*col],y[row*col]; 
	vector<float> featureVec;
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
	featureVec.push_back(xMean);
	featureVec.push_back(yMean);

	auto u=[N,xMean,yMean](const int p,const int q,const int* x,const int* y){
		float sum=0,tmpX,tmpY,pro;
		for(int i=0;i<N;i++){
			tmpX=x[i]-xMean;
			tmpY=y[i]-yMean;
			pro=1;
			for(int j=0;j<p;j++) pro*=tmpX;
			for(int j=0;j<q;j++) pro*=tmpY;
			sum+=pro;
		}	
		for(int i=0;i<p+q;i++){
			sum/=N;
		}
		return sum;
	};
	featureVec.push_back(u(1,1,x,y));
	featureVec.push_back(u(0,2,x,y));
	featureVec.push_back(u(2,0,x,y));

	float theta=0.5*atan2(2*featureVec[2],featureVec[3]-featureVec[4]);
	float vx=cos(theta),vy=sin(theta);
	float maxX=0,maxY=0,minX=10000,minY=10000;
	float tmpX,tmpY;
	for(int i=0;i<N;i++){
		x[i]-=xMean;
		y[i]-=yMean;
		tmpX=vx*x[i]-vy*y[i];
		tmpY=vy*x[i]+vx*y[i];
		if(maxX<tmpX) maxX=tmpX;
		if(tmpY>maxY) maxY=tmpY;
		if(tmpX<minX) minX=tmpX;
		if(tmpY<minY) minY=tmpY;
	}
	tmpX=(maxX+minX)/2;
	tmpY=(maxY+minY)/2;
	xMean+=vx*tmpX+vy*tmpY;
	yMean+=vx*tmpY-vy*tmpX;
	featureVec[0]-=xMean;
	featureVec[1]-=yMean;
	tmpX=maxX-minX;
	tmpY=maxY-minY;
	ans=RotatedRect(Point2f(xMean,yMean),Size2f(tmpX,tmpY),-theta*180.0/M_PI);
	featureVec.push_back(N/(tmpX*tmpY));
	featureVec.push_back(tmpY/tmpX);
	if(savePath!=""){
		char* savePath_=new char[savePath.size()+1];
		char* label_=new char[label.size()+1];
		strcpy(label_,label.c_str());
		strcpy(savePath_,savePath.c_str());
		append_image_data_csv(savePath_,label_,featureVec);
		delete[] savePath_;
		delete[]label_;
	}
	return featureVec;
}
vector<float> getDnnFeature(Mat &map, RotatedRect &rRect){
	vector<Point2f> points;
	rRect.points(points);
	float minX=1000,minY=1000,maxX=0,maxY=0;
	for(int i=0;i<4;i++){
		if(points[i].x<minX) minX=points[i].x;
		if(points[i].x>maxX) maxX=points[i].x;
		if(points[i].y<minY) minY=points[i].y;
		if(points[i].y>maxY) maxY=points[i].y;
	}
	Rect rect(Point(minX,minY),Point(maxX,maxY));
	string tmp="NetWorkThings/or2d-normmodel-007.onnx";
	char* tmp_=new char[tmp.size()+1];
	strcpy(tmp_,tmp.c_str());
	dnn::Net net=dnn::readNet(tmp_);
	Mat embedding;

	getEmbedding(map,embedding,rect,net,0);
	
	return embedding;

}

int drawBox(RotatedRect &box,vector<float> features,Mat &dst){
	if(dst.rows<=0||dst.cols<=0){
		cout<<"expecting non-empty image"<<endl;
	}
	Point2f vertices[4];
	box.points(vertices);
	Scalar s;
	if(dst.channels()==1)s=Scalar(255);
	else if(dst.channels()==3) s=Scalar(255,0,255);
	
	for(int i=0;i<4;i++){
		line(dst,vertices[i],vertices[(i+1)%4],s);
	}
	string text;
	text=to_string(features[3]);
	//for(float f:features){
	//	text+=to_string(f)+" ";
	//}
	putText(dst,text,vertices[0],FONT_HERSHEY_SIMPLEX,0.5,s,2);
	return 0;
}

