#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "features.h"
#include "readfile.h"
using namespace std;
using namespace cv;
vector<float> mat2vec(Mat src){
	if(src.type()!=CV_32F) src.convertTo(src,CV_32F);
	Mat tmp=src.reshape(1,mat.total());
	return vector<float>(tmp.begin<float>(),tmp.end<float>());
}
int main(int argc, char ** argv){

	vector<string> images;
	listImgs(argv[1],images);
	Mat img,feature;
	for (string s:images){
		img=imread(s);
		central7x7(img,feature);
		append_image_data_csv(imgCsv, s,mat2vec(feature));
	}	
	return 0;
} 
