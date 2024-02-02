#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(){
	string s="cathedral.jpeg";
	Mat img=imread(s,IMREAD_COLOR);
	imshow("img",img);
	waitKey(0);
	return 0;
}	
