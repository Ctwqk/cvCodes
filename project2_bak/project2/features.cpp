#include "features.h"
#include "../project1/task2/filter.h"
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
}*/

vector<float> mat2vec(Mat &src){
	if(src.type()!=CV_32F) src.convertTo(src,CV_32F);
	vector<float> ans;
	ans.assign((float*)src.datastart,(float*)src.dataend);
	//return vector<float>(tmp.begin<float>(),tmp.end<float>());
	return ans;
}

int central7x7(Mat &src,Mat &feature){
	int row=src.rows,col=src.cols;
	if(row<7||col<7){
		cout<<"expected size > 7x7"<<endl;
		return -1;
	}
	feature=src(Rect(col/2-3,row/2-3,7,7)).clone();
	imshow("feature",feature);	
	return 0;
}

int histogram(Mat &src,vector<float> &feature){ //R&G
	int row=src.rows,col=src.cols;
	float x,y,sum;
	if(src.type()!=CV_32FC3) src.convertTo(src,CV_32FC3);
	feature=vector<float>(32*32,0);
	for(int i=0;i<row;i++){
		float* cptr=src.ptr<float>(i);
		for(int j=0;j<col;j++){
			sum=cptr[3*j]+cptr[3*j+1]+cptr[3*j+2];
			sum?x=cptr[3*j+1]/sum*31+0.5:x=0;
			sum?y=cptr[3*j+2]/sum*31+0.5:y=0;	
			feature[(int)x*32+(int)y]++;
		}
	}
	return 0;
	
}

int multiHist(Mat &src,vector<float> &feature){ //4 corners and center
	int row=src.rows, col=src.cols;
	Mat tmp=src(Rect(col/2-10,row/2-10,21,21)).clone();
	vconcat(tmp,src(Rect(0,row-21,21,21)).clone(),tmp);
	vconcat(tmp,src(Rect(col-21,0,21,21)).clone(),tmp);
	vconcat(tmp,src(Rect(col-21,row-21,21,21)).clone(),tmp);
	vconcat(tmp,src(Rect(0,0,21,21)).clone(),tmp);
	histogram(tmp,feature);
	return 0;
}

int texture(Mat &src,vector<float> &feature){
	int row=src.rows,col=src.cols;
	float min=1000000,max=-1000000;
	if(src.type()!=CV_32FC3) src.convertTo(src,CV_32FC3);
	Mat sx,sy,mag=Mat::zeros(row,col,CV_32FC3),ang=Mat::zeros(row,col,CV_32FC3);
	Sobel(src,sx,CV_32F,1,0,3);
	Sobel(src,sy,CV_32F,0,1,3);
	feature=vector<float>(32*25,0);
	for(int i=0;i<row;i++){
		const float* xPtr=sx.ptr<float>(i);
		const float* yPtr=sy.ptr<float>(i);
		float* aPtr=ang.ptr<float>(i);
		float* mPtr=mag.ptr<float>(i);
		for(int j=0;j<col;j++){
			for(int k=0;k<3;k++){
				int tmp=3*j+k;
				aPtr[tmp]=180+atan2(yPtr[tmp],xPtr[tmp])*180/CV_PI;
				mPtr[tmp]=sqrt(yPtr[tmp]*yPtr[tmp]+xPtr[tmp]*xPtr[tmp]);
				if(min>mPtr[tmp]) min=mPtr[tmp];
				if(max<mPtr[tmp]) max=mPtr[tmp];
			}
		}
	}
	max-min?mag=mag/(max-min):mag=Mat::zeros(row,col,CV_32FC3);
	for(int i=0;i<row;i++){
		const float* aPtr=ang.ptr<float>(i);
		const float* mPtr=mag.ptr<float>(i);
		for(int j=0;j<col;j++){
			for(int k=0;k<3;k++){
				int tmp=3*j+k;
				feature[int(aPtr[tmp]/15)*32+int(mPtr[tmp]*31+0.5)]++;
			}
		}
	}		
	return 0;
}

int searchCsv(string imgsName, string tarName, vector<float> &feature){
	char* imgsName_=new char[imgsName.size()+1];
	strcpy(imgsName_,imgsName.c_str());
	vector<char* >fileNames;
	vector<vector<float>> data;
	read_image_data_csv(imgsName_,fileNames,data);
	for(int i=0;i<fileNames.size();i++){
		if(tarName.find(fileNames[i])!=string::npos){
			feature=data[i];
			return 0;
		}
	}
	cout<<"target not fount"<<endl;
	return -1;
}	
