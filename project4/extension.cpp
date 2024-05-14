/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 * this function implements recognizing multiple targets(chessboard) and make the target looks not like chessboard
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
using namespace std;
using namespace cv;

int main(int argc,char** argv){
	Mat replace=imread("smile.jpeg");// image to replace the chessboard
	if(replace.empty()){
		cout<<"failed to read the replacement image"<<endl;
		return -1;
	}
	ifstream inStream("intcPara.txt");//read in intrinsic parameters of the camera
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
	vector<Point2f> replaceCorners={Point2f(0,0),Point2f(replace.cols,0),Point2f(replace.cols,replace.rows),Point2f(0,replace.rows)};
	Mat src,frame,rvec,tvec,mask,tmp,background,mask_inv;
	bool flag,foundCornor;
	vector<Point2f> corner_set;
	Size patternsize(x,y);
	vector<vector<Point2f>> corner_list;
	vector<vector<Point3f>> point_list;
	vector<Point3f> point_set;
	vector<vector<Point2f>> masks;
	vector<Point2f> paraf;
	vector<Point> para;
	vector<Point3f> targets={Point3f(0,0,3),Point3f(0,1,3),Point3f(0,0,4),Point3f(1,0,3)};
	vector<Point2f>results;
	vector<Scalar> scalars{Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0)};

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
		*capdev>>frame;
		cvtColor(frame,src,COLOR_BGR2GRAY);
		do{//loop when there are still target exist in the grayscale image
			foundCornor=findChessboardCorners(src,patternsize,corner_set);
			if(foundCornor){
				cornerSubPix(src,corner_set,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,30,0.1));
				solvePnP(point_set,corner_set,camera_matrix,distCoef,rvec,tvec);
				//every time a target is founded, in the color frame cover it with replace image, and black it out in he gray scale image
				paraf={corner_set[0],corner_set[x-1],corner_set.back(),corner_set[x*y-x]};
				para=vector<Point>(paraf.size());
				transform(paraf.begin(),paraf.end(),para.begin(),[](const Point2f&pt){return Point(static_cast<int>(pt.x),static_cast<int>(pt.y));});
				mask=Mat::zeros(src.size(),src.type());
				fillConvexPoly(src,para,Scalar(0));//black out in gray scale image
				warpPerspective(replace,tmp,getPerspectiveTransform(replaceCorners,paraf),frame.size());//calculate the perspective transform matrix and do transformation to the raplace image, the result will be stored in `tmp`
				fillConvexPoly(mask,para,Scalar(255));//mask the ROI, where the target is
				bitwise_not(mask,mask_inv);
				background=Mat::zeros(frame.size(),CV_8UC1);//background is the image that target is cut off
				frame.copyTo(background,mask_inv);
				add(tmp,background,frame,mask);// the cut off target will be replaced by the given replacing image
				
				//drawing 3 axis to prove that the target is founded
				projectPoints(targets,rvec,tvec,camera_matrix,distCoef,results);
				circle(frame,results[0],5,Scalar(255,0,255),-1);
				for(int i=1;i<4;i++){
					line(frame,results[0],results[i],scalars[i-1],3,LINE_4);
				}
			}	
		
		}	while(foundCornor);
		imshow("vid",frame);
	}

	return 0;
}
