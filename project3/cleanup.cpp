#include "cleanup.h"
using namespace std;
using namespace cv;
int growing(Mat &src, Mat &kernel,Mat & dst){
	if(src.channels()>1){
		cout<<"expecting binary image"<<endl;
		return -1;
	}
	int row=src.rows, col=src.cols;
	int r=kernel.rows,c=kernel.cols;
	dst=Mat::zeros(row,col,CV_8U);
	uchar** data=new uchar*[row];
	for(int i=0;i<row;i++){
		data[i]=dst.ptr<uchar>(i);
	}
	for(int i=0;i<row;i++){
		uchar* ptr=src.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			for(int k=-r/2;k<r/2;k++){
				for(int l=-c/2;l<c/2;l++){

				
	
int denoise(Mat &src, Mat &dst){
	int row=src.rows, col=src.cols;

	
