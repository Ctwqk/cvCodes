#include <iostream>
#include "region.h"
using namespace std;
using namespace cv;
int main(){
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
	return 0;
}
