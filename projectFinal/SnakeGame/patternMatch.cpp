/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * given the patterns from a dataset, match current frame with one of them.
 */
#include "patternMatch.hpp"
#include <opencv2/xfeatures2d.hpp>
using namespace std;
using namespace cv;


int matchPattern(vector<cv::Mat> &patternDescriptors, Mat &queryDescriptor,vector<vector<KeyPoint>> &patternKeypoints, vector<KeyPoint> &queryKeypoints,BFMatcher &matcher,vector<DMatch> &matches,int &tar,Mat &homography_matrix,int idx){

	/*Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400);
	Mat img=imread("card.webp",IMREAD_GRAYSCALE);
	Mat desc;
	vector<KeyPoint> kep;
	detector->detectAndCompute(img,Mat(),kep,desc);
	*/
	Mat img=imread("card.webp",IMREAD_GRAYSCALE);
	Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400);
	
	int n=patternDescriptors.size();
	vector<DMatch> final_matches;
	if(idx<0||idx>=n){
		//unfinished
		vector<DMatch> tmp_match;
		matcher.match(queryDescriptor,final_matches);
		int * matchNum = new int[n]();
		for(int i=0;i<n;i++){
			matcher.match(patternDescriptors[i],queryDescriptor,tmp_match);
			matchNum[i]+=tmp_match.size();
		}		
		tar=distance(matchNum,max_element(matchNum,matchNum+n));
		
	}
	else{
		tar=idx;
		vector<KeyPoint> kp;
		//detector->detectAndCompute(img,Mat(),kp,patternDescriptors[idx]);
		//cout<<kp.size()<<endl;
		matcher.match(patternDescriptors[idx],queryDescriptor, final_matches);
	}
	if(getHomoMat(patternKeypoints[tar],queryKeypoints,final_matches,homography_matrix)){
		return -1;
	}
	matches=final_matches;
	
	return 0;
}

int getHomoMat(vector<KeyPoint> &patternKeypoints,vector<KeyPoint> &queryKeypoints,vector<DMatch> &matches,Mat &homography_matrix){
	vector<Point2f> trainPoints, queryPoints;
	for(DMatch match: matches){
		queryPoints.push_back(queryKeypoints[match.queryIdx].pt);
		trainPoints.push_back(patternKeypoints[match.trainIdx].pt);
	}
	Mat mask;
	//cout<<"embedded "<<queryPoints.size()<<" "<<trainPoints.size()<<endl;
	if(!trainPoints.size()){
		return -1;
	}	
	homography_matrix=findHomography(queryPoints,trainPoints,RANSAC,3,mask);
	//cout<<homography_matrix.size()<<endl;
	if(homography_matrix.size()==Size(0,0)){
		return -1;
	}
	vector<DMatch> tmp_match;
	const uchar* ptr=mask.ptr<uchar>(0);
	for(size_t i=0;i<mask.rows;i++){
		if(ptr[i]){
			tmp_match.push_back(matches[i]);
		}
	}
	matches=tmp_match;
	return 0;
}

int getHomoMat(vector<Point2f> &patternpoints,vector<Point2f> &querypoints,Mat &homography_matrix){
	Mat mask;
	if(!patternpoints.size()){
		return -1;
	}	
	homography_matrix=findHomography(querypoints,patternpoints,RANSAC,3,mask);
	return 0;
}
