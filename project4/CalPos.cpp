/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 * this file draws a virtual 5-point star above the target
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
using namespace std;
using namespace cv;
int main(int argc,char** argv){
	ifstream inStream("intcPara.txt");
	if(!inStream){
		cout<<"failed to read from file"<<endl;
		return -1;
	}
	Mat camera_matrix(3,3,CV_64F);	
	vector<double> distCoef;
	int x,y;
	inStream>>x;
	inStream>>y;
	double *p=camera_matrix.ptr<double>(0);
	for(int i=0;i<9;i++){
		inStream>>p[i];
	}
	double value;
	while(inStream>>value){
		distCoef.push_back(value);
	}
	/*
	cout<<camera_matrix<<endl;
	for(double d:distCoef) cout<<d<<" ";
	cout<<endl;
	*/
	VideoCapture *capdev=new VideoCapture(atoi(argv[1]));
	if(!capdev->isOpened()){
		cout<<"failed to open the camera"<<endl;
		return -1;
	}
	Mat src,frame,rvec,tvec;
	bool flag,foundCornor;
	vector<Point2f> corner_set;
	Size patternsize(x,y);
	vector<vector<Point2f>> corner_list;
	vector<vector<Point3f>> point_list;
	vector<Point3f> point_set;
	vector<Point3f> targets={Point3f(0,0,3),Point3f(0,1,3),Point3f(0,0,4),Point3f(1,0,3)};//these are used to draw axis
	vector<Point2f>results;
	vector<Scalar> scalars{Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0)};
	vector<Point3f> targets2={//this is the points of a 5-point star
			Point3f(4,0,1),
			Point3f(6.8532,-1.9271,1),
			Point3f(5.7642,-5.4271,1),
			Point3f(2.2366,-5.4271,1),
			Point3f(1.1468,-1.9271,1),
			Point3f(4,-3,2),
			Point3f(4,-3,0)
	};
	vector<Point2f> results2;
	for(int i=0;i<y;i++){
		for(int j=0;j<x;j++){
			point_set.push_back(Vec3f(j,-i,0));
		}
	}
	for(;;){
		char c=waitKey(3);
		if(c=='q'){
			destroyAllWindows();
			waitKey(1);
			break;
		}
		flag=0;
		*capdev>>frame;
		cvtColor(frame,src,COLOR_BGR2GRAY);
		foundCornor=findChessboardCorners(src,patternsize,corner_set);
		if(foundCornor){
		    cornerSubPix(src,corner_set,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,30,0.1));
			flag=1;
			solvePnP(point_set,corner_set,camera_matrix,distCoef,rvec,tvec);
		}
		for(Point2f p:corner_set){
			/*
			idx++;
			putText(frame,to_string(idx),p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
			*/
			circle(frame, p, 1, Scalar(0,0,255),-1);
		}
		//putText(frame,"rot",Point2f(0,20),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
		if(foundCornor) {
			double* rp=rvec.ptr<double>(0);
			double* tp=tvec.ptr<double>(0);
			//these codes put the value of rotation and translation at the topleft corner of the window
			putText(frame,"rot:",Point2f(0,20),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
			putText(frame,"trs:",Point2f(0,80),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
			for(int i=0;i<rvec.size().height;i++){
				putText(frame,to_string((float)rp[i]),Point2f(35,20+i*20),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
				putText(frame,to_string((float)tp[i]),Point2f(35,80+i*20),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
			}
			//cout<<rvec;
			//cout<<tvec<<endl;
			projectPoints(targets,rvec,tvec,camera_matrix,distCoef,results);
			circle(frame,results[0],5,Scalar(255,0,255),-1);
			for(int i=1;i<4;i++){
				line(frame,results[0],results[i],scalars[i-1],3,LINE_4);
			}
			projectPoints(targets2,rvec,tvec,camera_matrix,distCoef,results2);
			for(int i=0;i<5;i++){
				line(frame,results2[i],results2[5],Scalar(0,0,255),3,LINE_4);
				line(frame,results2[i],results2[6],Scalar(0,0,255),3,LINE_4);
				for(int j=i+2;j<5;j++){
					if(i==0&&j==4) continue;
					line(frame,results2[i],results2[j],Scalar(0,255,255),3,LINE_4);
				}
			}	
		}
		imshow("vid",frame);
	}
	return 0;
}
