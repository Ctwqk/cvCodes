/*
 * Author:  Taiwei Cui
 * Time: 2024-3-18
 * This file implement the camera calibration. The camera_matrix and distcoeff and the pattern size will be restored in the file intcPara.txt.
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
using namespace std;
using namespace cv;
int main(int argc, char** argv){
	int MIN_NUM=5;//only when at least 5 images are chosen will the system calculate the intrinsic parameters
	VideoCapture *capdev;
	capdev=new VideoCapture(atoi(argv[1]));
	if(!capdev->isOpened()){
		cout<<"failed to open the camera"<<endl;
		return -1;
	}
	Mat frame,src,camera_matrix;
	vector<Point2f> corner_set;//store the 2-D coordinate of corner pixels
	Size patternsize(9,6);
	vector<vector<Point2f>> corner_list;// store the 2-D corner vectors of chosen frames
	vector<vector<Point3f>> point_list;// store the 3-D corner vectors of chosen frames
	vector<Point3f> point_set;// store the 3-D coordinate of corner pixels
	double error;
	
	vector<double> distCoef;//initialize the distortion matrix
	Mat rvec,tvec;// store the rotation and transation
	double data[9]={1,0,frame.cols/2.0,0,1,frame.rows/2.0,0,0,1};//data for initialize the camera matrix
	for(int i=0;i<6;i++){
		for(int j=0;j<9;j++){
			point_set.push_back(Vec3f(j,-i,0));
		}
	}
	bool foundCornor,flag;
	for(;;){
		char c=waitKey(10);
		if(c=='q'){
			destroyAllWindows();
			waitKey(1);
			break;
		}
		flag=0;
		*capdev>>frame;
		data[2]=frame.cols/2.0;
		data[5]=frame.rows/2.0;
		camera_matrix=Mat(3,3,CV_64F,data);

		cvtColor(frame,src,COLOR_BGR2GRAY);
		foundCornor=findChessboardCorners(src,patternsize,corner_set);
		if(corner_set.size()!=0){
		    cornerSubPix(src,corner_set,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,30,0.1));
			drawChessboardCorners(src,patternsize,Mat(corner_set),foundCornor);
			flag=1;
		}
		if(c=='s'&&flag){//when s is pressed and corners are found, the intrinsic parameters and error will be calculated
			corner_list.push_back(corner_set);
			point_list.push_back(point_set);
			if(corner_list.size()>=MIN_NUM){
				rvec=(0,0,0);
				tvec=(0,0,0);
				cout<<"init camera matrix: "<<endl<<camera_matrix<<endl;
				//cout<<point_list[4].size()<<" "<<corner_list[4].size()<<endl;
				error=calibrateCamera(point_list,corner_list,src.size(),camera_matrix,distCoef,rvec,tvec,CALIB_FIX_ASPECT_RATIO);

//				error=calibrateCamera(point_list,corner_list,src.size(),camera_matrix_,distCoef,rvec,tvec,CALIB_RATIONAL_MODEL);
				cout<<"camera matrix: "<<endl<<camera_matrix<<endl<<"distortion coef: "<<endl;
				for(auto i:distCoef) cout<<i<<" ";
				cout<<endl;
				cout<<"error: "<<error<<endl;
			}
		}
		//int idx=0;
		for(Point2f p:corner_set){//drawing the corners
			/*
			idx++;
			putText(frame,to_string(idx),p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
			*/
			circle(frame, p, 3, Scalar(0,0,255),-1);
		}
		imshow("cor",frame);
	}	
	if(corner_list.size()>=MIN_NUM){//if intrinsic parameters have been calculated, asking user to determing whether to write them to file
		cout<<distCoef.size()<<" "<<MIN_NUM<<endl;
		cout<<"save intrinsic parameters to file?[Y/N]"<<endl;
		char c=getchar();
		if(c=='Y'){
			//ofstream outStream("intcPara.txt",ios::app);
			ofstream outStream("intcPara.txt");
			if(!outStream){
				cout<<"failed to write to file"<<endl;
				return -1;
			}
			outStream<<patternsize.width<<" ";
			outStream<<patternsize.height<<endl;
			double *p=camera_matrix.ptr<double>(0);
			for(int i=0;i<9;i++){
				outStream<<p[i]<<" ";
			}
			outStream<<endl;
			for(double d:distCoef){
				outStream<<d<<" ";
			}
			outStream<<endl;
		}
	}
	return 0;
}
