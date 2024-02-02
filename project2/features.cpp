#include "features.h"
/*
vector<float> mat2vec(Mat &src){
	if(src.type()!=CV_32F) src.convertTo(src,CV_32F);
	Mat tmp=src.reshape(1,src.total());
	return vector<float>(tmp.begin<float>(),tmp.end<float>());
}
*/
/*
Mat feature central7x7(Mat &src){
	int row=src.rows,col=src.cols;
	if(row<7||col<7){
		cout<<"expected size > 7x7"<<endl;
		return -1;
	}
	feature=src(Rect(col/2-25,row/2-25,51,51)).clone();
	imshow("src",src);
	imshow("feature",feature);	
	return 0;
}
*/
/*
vector<vector<float>> getFeatures(Mat &src){
	vector<Mat> ans;
	ans.push_back(mat2vec(central7x7(src));

	return ans;
*/}
int feature central7x7(Mat &src,Mat &feature){
	int row=src.rows,col=src.cols;
	if(row<7||col<7){
		cout<<"expected size > 7x7"<<endl;
		return -1;
	}
	feature=src(Rect(col/2-25,row/2-25,51,51)).clone();
	imshow("src",src);
	imshow("feature",feature);	
	return 0;
}
