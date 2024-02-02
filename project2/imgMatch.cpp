#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int argc,char **argv){
	Mat	img=argv[1];
	Mat feature;
	central7x7(img,feature);
	
	vector<vector<float>> features;
	vector<char *> fileNames;
	read_image_data_csv("imgCsv", 	fileNames, features);
	vector<pair<float, int>> rank;
	for(int i=0;i<filenames.size();i++){
		rank.push_back(dist(features[i],feature),i);
	}
	sort(rank.begin(),rank.end(),greater<float>());
	fig1=imread(rank[0],IMREAD_COLOR);
	fig2=imread(rank[0],IMREAD_COLOR);
	fig3=imread(rank[0],IMREAD_COLOR);
	imshow("fig1", fig1);
	imshow("fig2", fig2);
	imshow("fig3", fig3);
	waitKey(0);
	return 0;

	
}
	
