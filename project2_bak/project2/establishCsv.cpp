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

	vector<char* > images;
	listImgs(argv[1],images);
	Mat img,feature;
	char s[256];
	int idx=0;
	for (int i=0;i<images.size();i++){
		strcpy(s,images[i]);
		for(int j=0;j<256;j++){
			if(s[j]=='\0') break;
			printf("%c", s[j]);
		}
		printf("%d \n",idx++);
		img=imread(s);
		central7x7(img,feature);
		vector<float> tmp=mat2vec(feature);
		append_image_data_csv("imgCsv", s,tmp);
	}	
	return 0;
} 
