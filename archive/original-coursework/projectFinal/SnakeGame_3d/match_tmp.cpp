/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * match frame with pattern image. It is called tmp because it need some improvement, and i am still working on that .
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "saveOrReadPattern.hpp"
#include "patternMatch.hpp"

using namespace std;
using namespace cv;
int main(int argc,char** argv){
	if(argc<3){
		cout<<"usage: ./main <camera index> <path to pattern file>"<<endl;
		return -1;
	}

	Mat debugMatch,debugImg=imread("card.webp",IMREAD_GRAYSCALE);
	

	vector<vector<KeyPoint>> keypointsOfPatterns;
	vector<Mat> descriptorsOfPatterns;
	vector<vector<Point3f>> point3ds;
	vector<Mat> mat_3ds;
	Mat mat_3d_invid;
	vector<Point2f> results(4,Point2f(0,0));
	vector<Point3f> targets={Point3f(0,0,1),Point3f(0,1,1),Point3f(0,0,2),Point3f(1,0,1)};//these are used to draw axis
	readPatternsFromCsv(argv[2],keypointsOfPatterns,descriptorsOfPatterns,mat_3ds,point3ds);
	drawKeypoints(debugImg,keypointsOfPatterns[0],debugMatch,Scalar::all(-1));
	imshow("check",debugMatch);
	cout<<"size"<<keypointsOfPatterns[0].size()<<endl;
	waitKey(0);
		
	int minHessian=400;
	vector<KeyPoint> keypoints;
	Mat descriptor;
	vector<DMatch> matches;
	BFMatcher matcher;
	matcher.add(descriptorsOfPatterns);
	matcher.train();
	Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);
	int matchResult;
	vector<Point2f> queryPoints;
	vector<Point2f> trainPoints;
	//detector->detectAndCompute(debugImg,Mat(),keypointsOfPatterns[0],descriptorsOfPatterns[0]);
	//cout<<keypointsOfPatterns[0].size()<<endl;
	//cout<<3<<endl;

	int x,y;
	ifstream inStream("intcPara.txt");
	inStream>>x;
	inStream>>y;
	vector<double> distCoef;
	Mat camera_matrix(3,3,CV_64F);
	VideoCapture *vidCap=new VideoCapture(atoi(argv[1]));
	if(!vidCap->isOpened()){
		cout<<"failed to open the camera "<<argv[1]<<endl;
		return -1;
	}
	Mat frame,gray_frame,gray_prev;
	Mat rvec,tvec;
	vector<Point2f> points_2d={4,Point2f(0,0)};
	Mat points_3d_res_mat;
	vector<Scalar> scalars{Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0)};
	bool isFollowing=false,solved=false;
	Size winSize(21,21);
	int maxLevel=3;
	TermCriteria criteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
	vector<uchar> trackingStatus;
	int trackingSuccNum=0;
	vector<float> err;	
	float maxAcceptableError=10.;
	Mat homography_matrix;
	Mat mask;
	//cout<<5<<endl;
	

	vector<Point2f> trackingPoints;
	vector<Point2f> trackingPoints_pattern;
	for(;;){
		char c=waitKey(5);
		if(c=='q'){
			destroyAllWindows();
			waitKey(1);
			break;
		}
		*vidCap>>frame;
		cvtColor(frame,gray_frame,COLOR_BGR2GRAY);
		if(isFollowing){
			//cout<<"tracked "<<endl;
			calcOpticalFlowPyrLK(gray_frame,gray_prev,trackingPoints, trackingPoints,trackingStatus, err,winSize,maxLevel,criteria);
			trackingSuccNum=0;
			vector<Point2f> newTrackingPoints;
			vector<Point2f> newTrackingPoints_pattern;
			
			for(int i=0;i<err.size();i++){
				if(trackingStatus[i]==1){
					newTrackingPoints.push_back(trackingPoints[i]);
					newTrackingPoints_pattern.push_back(trackingPoints_pattern[i]);
					
					if(err[i]<maxAcceptableError){
						trackingSuccNum++;
					}
				}
			}
			trackingPoints_pattern=newTrackingPoints_pattern;
			trackingPoints=newTrackingPoints;
			if(trackingSuccNum>matches.size()/4){
				isFollowing=1;
				cout<<"track "<<trackingPoints.size()<<" "<<trackingPoints_pattern.size()<<endl;
				if(trackingPoints.size()<4){
					isFollowing=0;
				}
				else homography_matrix=findHomography(trackingPoints,trackingPoints_pattern,RANSAC,3,mask);
				cout<<"track "<<homography_matrix.size()<<endl;
				if(homography_matrix.size()==Size(0,0)){
					isFollowing=0;
				}
			}
			else{
				isFollowing=0;
			}
		}
		//cout<<"again"<<endl;
		
		if(!isFollowing){
			trackingPoints.clear();
			trackingPoints_pattern.clear();
			cout<<"detected "<<endl;
			detector->detectAndCompute(gray_frame,Mat(),keypoints, descriptor);	
			if(matchPattern(descriptorsOfPatterns,descriptor,keypointsOfPatterns,keypoints,matcher,matches,matchResult,homography_matrix,0)==0){
				for(DMatch match:matches){
					trackingPoints_pattern.push_back(keypointsOfPatterns[matchResult][match.trainIdx].pt);
					trackingPoints.push_back(keypoints[match.queryIdx].pt);
				}
				isFollowing=1;
				//cout<<"match "<<homography_matrix.size()<<endl;
			}
		}

		cout<<isFollowing<<" "<<matchResult<<endl;
		if(isFollowing){
			solved=false;
			cout<<homography_matrix.size()<<" "<<mat_3ds[matchResult].size()<<endl;
			mat_3d_invid=homography_matrix*mat_3ds[matchResult];
			const double* fptr1=mat_3d_invid.ptr<double>(0);
			const double* fptr2=mat_3d_invid.ptr<double>(1);
			for(int i=0;i<4;i++){
				points_2d[i].x=fptr1[i];
				points_2d[i].y=fptr2[i];
			}
			
			solved=solvePnP(point3ds[matchResult],points_2d,camera_matrix,distCoef,rvec,tvec,false,SOLVEPNP_P3P);
			if(solved){
				cout<<"projecting"<<endl;
				projectPoints(targets,rvec,tvec,camera_matrix,distCoef,results);
				circle(frame,results[0],5,Scalar(255,0,255),-1);
				for(int i=1;i<4;i++){
					line(frame,results[0],results[i],scalars[i-1],3,LINE_4);
				}
			}
			
		}
		Mat tmpdescp;
		vector<KeyPoint> tmpkep;
		//detector->detectAndCompute(debugImg,Mat(),tmpkep,tmpdescp);
		//matcher.match(tmpdescp,descriptor,matches);
		//cout<<keypointsOfPatterns.size()<<endl;
		drawMatches(debugImg,keypointsOfPatterns[0],frame,keypoints,matches,debugMatch,Scalar::all(-1), Scalar::all(-1), vector<char>());
		//cout<<"hhh"<<tmpdescp-descriptorsOfPatterns[0]<<endl;
		//drawMatches(debugImg,keypointsOfPatterns[0],frame,keypoints,matches,debugMatch,Scalar::all(-1), Scalar::all(-1), vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("vid",debugMatch);
		gray_prev=gray_frame;
		
		
	}
	return 0;
}
