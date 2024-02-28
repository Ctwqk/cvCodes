/*
 * Author: Taiwei Cui
 * Time: Feb 27 2024
 *
 * */

#include "features.h"
using namespace std;
using namespace cv;

// map:  the region map
// rRect: the RotatedRect that contains the target region
// return the embedding of dnn
// this funtion is largely based on the dnnexample.cpp from Bruce A. Maxwell
vector<float> getDnnFeature(Mat &map, RotatedRect &rRect){
	vector<Point2f> points;
	rRect.points(points);
	//find the smallest cv::Rect that contains the given RotatedRect
	float minX=map.cols,minY=map.rows,maxX=0,maxY=0;
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

//this function will calculate and return the feature vector of given image map //map: given image map
//ans: hold the oriented bounding box of the region
//savePath: the CSV file in which saves the feature vector ( if is given)
//label: the label to which the region belong ( if is given)
vector<float> featureBox(Mat &map,RotatedRect &ans,string savePath,string label){
	int N=0,row=map.rows,col=map.cols;
	float xSum=0,ySum=0,xMean,yMean;
	int x[row*col],y[row*col]; 
	vector<float> featureVec;
	//calculate how many pixels in the item
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
	featureVec.push_back(xMean);//featureVec is used to hold this value by now, this value is not in the final feature vector
	featureVec.push_back(yMean);//featureVec is used to hold this value by now, this value is not in the final feature vector

	//this lambda function calculate the u_pq(xy)
	//p: index for x
	//q: index for y
	//x & y: array of int
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
	featureVec.push_back(u(1,1,x,y));//featureVec is used to hold this value by now, this value is not in the final feature vector
	featureVec.push_back(u(0,2,x,y));//featureVec is used to hold this value by now, this value is not in the final feature vector
	featureVec.push_back(u(2,0,x,y));//featureVec is used to hold this value by now, this value is not in the final feature vector

	//axis with least moment
	float theta=0.5*atan2(2*featureVec[2],featureVec[3]-featureVec[4]);
	float vx=cos(theta),vy=sin(theta);

	float maxX=0,maxY=0,minX=10000,minY=10000;
	float tmpX,tmpY;
	for(int i=0;i<N;i++){
		x[i]-=xMean;
		y[i]-=yMean;
		tmpX=vx*x[i]-vy*y[i]; //tmpX and tmpY hold the cooperate in new basis.
		tmpY=vy*x[i]+vx*y[i];
		if(maxX<tmpX) maxX=tmpX;
		if(tmpY>maxY) maxY=tmpY;
		if(tmpX<minX) minX=tmpX;
		if(tmpY<minY) minY=tmpY;
	}
	tmpX=(maxX+minX)/2;
	tmpY=(maxY+minY)/2;
	//calculate the central point in old basis
	xMean+=vx*tmpX+vy*tmpY;
	yMean+=vx*tmpY-vy*tmpX;
	featureVec[0]-=xMean;
	featureVec[1]-=yMean;
	tmpX=maxX-minX;
	tmpY=maxY-minY;
	ans=RotatedRect(Point2f(xMean,yMean),Size2f(tmpX,tmpY),-theta*180.0/M_PI);
	//three feature that is used in feature vector
	featureVec.push_back(N/(tmpX*tmpY));
	featureVec.push_back(tmpX/tmpY);
	featureVec.push_back(sqrt(N)/(tmpX+tmpY));
	
	featureVec.erase(featureVec.begin(),featureVec.end()-3);
	//if want to use dnn embedding, uncomment these lines
	//vector<float> embedding=getDnnFeature(map,ans);
	//featureVec.insert(featureVec.end(),embedding.begin(),embedding.end());
 	//if savePath and label is given, save it.
	if(savePath!=""||label!=""){
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

//this function draw the orientated box and the axis with minimal moment
//box: the oriented bounding box
//features: the features to be show
//dst: the image that bounding box,axis and text will be painted on
//name: the label of this region
int drawBox(RotatedRect &box,vector<float> features,Mat &dst,string name){
	if(dst.rows<=0||dst.cols<=0){
		cout<<"expecting non-empty image"<<endl;
	}
	Point2f vertices[4];
	box.points(vertices);
	Scalar s;
	if(dst.channels()==1)s=Scalar(255);
	else if(dst.channels()==3) s=Scalar(255,0,255);
	float angle=box.angle/180.0*M_PI;
	line(dst,box.center,Point2f(box.center.x+sin(angle)*30,box.center.y-cos(angle)*30),Scalar(255,255,255));
	
	for(int i=0;i<4;i++){
		line(dst,vertices[i],vertices[(i+1)%4],s);
	}
	string text;
	text=to_string(features.back());
	putText(dst,text,vertices[0],FONT_HERSHEY_SIMPLEX,0.5,s,2);
	if(name!="")putText(dst,name,vertices[3],FONT_HERSHEY_SIMPLEX,0.5,s,2);
	return 0;
}

