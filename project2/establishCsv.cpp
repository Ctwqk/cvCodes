#include <iostream>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "features.h"
#include "readfile.h"
using namespace std;
using namespace cv;
vector<float> mat2vec(Mat &src){
	if(src.type()!=CV_32F) src.convertTo(src,CV_32F);
	vector<float> ans;
	ans.assign((float*)src.datastart,(float*)src.dataend);
	//return vector<float>(tmp.begin<float>(),tmp.end<float>());
	return ans;
}
int main(int argc, char ** argv){

	vector<string > images;
	listImgs(argv[1],images);
	Mat img,feature;
	//char s[256];
	int idx=0;
	for (string s:images){
		img=imread(s);
		central7x7(img,feature);
		vector<float> tmp=mat2vec(feature);
		char* swa=new char[s.size()+1];	
		strcpy(swa,s.c_str());
		append_image_data_csv("imgCsv", swa,tmp);
		delete[] swa;
	}	
	return 0;
} 
