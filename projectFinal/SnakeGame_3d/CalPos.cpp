/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * this file draws a virtual 5-point star above the target
 */
#include "CalPos.hpp"
using namespace std;
using namespace cv;
int CalPos(Mat &frame,Mat &tvec,Mat &rvec,Mat &camera_matrix,vector<double> &distCoef,int x,int y){
	/*
	cout<<camera_matrix<<endl;
	for(double d:distCoef) cout<<d<<" ";
	cout<<endl;
	*/
	Mat src;
	bool flag=0,foundCornor=false;

	vector<Point2f> corner_set;
	Size patternsize(x,y);
	vector<vector<Point2f>> corner_list;
	vector<vector<Point3f>> point_list;
	vector<Point3f> point_set;
	for(int i=0;i<y;i++){
		for(int j=0;j<x;j++){
			point_set.push_back(Vec3f(j,-i,0));
		}
	}
		cvtColor(frame,src,COLOR_BGR2GRAY);
		foundCornor=findChessboardCorners(src,patternsize,corner_set);
		if(foundCornor){
		    cornerSubPix(src,corner_set,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,30,0.1));
			flag=1;
			solvePnP(point_set,corner_set,camera_matrix,distCoef,rvec,tvec);
		}
		else return -1;
		//putText(frame,"rot",Point2f(0,20),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
		/*if(foundCornor) {
			double* rp=rvec.ptr<double>(0);
			double* tp=tvec.ptr<double>(0);
			//these codes put the value of rotation and translation at the topleft corner of the window
			//cout<<rvec;
			//cout<<tvec<<endl;
			vector<vector<Point2f>> theResults;
			//cout<<targets.size()<<endl;
			vector<Point2f> result;
			for(vector<Point3f> target: targets){
				projectPoints(target,rvec,tvec,camera_matrix,distCoef,result);
				theResults.push_back(result);
			}
			results=theResults;
		}
		*/
	return 0;	
}
