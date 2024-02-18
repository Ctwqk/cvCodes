#include <iostream>
#include "region.h"
#include "denoise.h"
using namespace std;
using namespace cv;
int main(int argc,char** argv){
		
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
	regionize(mat,dst,2);
	
	cout<<endl;
	dst/=255;
	cout<<dst;

	Mat src,ans;
	src=imread(argv[1],IMREAD_GRAYSCALE);
	denoise(src,src);
	imshow("src",src);
	regionize(src,ans);
	imshow("reg",ans*4);
	mapToColor(ans,ans);
	imshow("clr",ans);	
	waitKey(0);
	return 0;
}
