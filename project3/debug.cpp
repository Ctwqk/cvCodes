#include <iostream>
#include "region.h"
#include "denoise.h"
#include "features.h"
#include "dist.h"
using namespace std;
using namespace cv;
int main(int argc,char** argv){
	/*	
	Mat mat=(Mat_<int>(9,7)<<0,0,0,0,0,0,0,
							 0,0,0,0,0,0,0,
							 0,0,1,1,0,0,0,
							 0,1,1,1,1,0,0,
							 0,1,1,1,1,0,0,
							 0,0,1,1,1,0,0,
							 0,0,0,1,0,0,0,
							 0,0,0,0,0,0,0,
							 0,0,0,0,0,0,1);
	cout<<mat<<endl;
	mat*=255;
	

	Mat dst;
	regionize(mat,dst);
	
	cout<<endl;
	dst/=255;
	cout<<dst;
*/
	Mat src,ans;
	src=imread(argv[1]);
	resize(src,src,Size(480,600));
	denoise(src,src);
	imshow("src",src);
	waitKey();
	vector<Mat> maps=regionize(src,ans);
	imshow("reg",ans);
	waitKey();
	RotatedRect rect;
	vector<float> features;
	int idx=0;
	for(auto a:maps){
		imshow(to_string(idx++),a);
		features=featureBox(a,rect);
		drawBox(rect,features,ans);
	}
	imshow("fin",ans);
	char c=waitKey(0);
	return 0;
}
