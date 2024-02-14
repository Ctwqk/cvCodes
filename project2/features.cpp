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
}*/

vector<float> mat2vec(Mat &src){  //convert a cv::Mat into std::vector<float>
	if(src.type()!=CV_32F) src.convertTo(src,CV_32F);
	vector<float> ans;
	ans.assign((float*)src.datastart,(float*)src.dataend);
	//return vector<float>(tmp.begin<float>(),tmp.end<float>());
	return ans;
}

int central7x7(Mat &src,Mat &feature){  //retrive the center 7x7 part of the Mat src
	int row=src.rows,col=src.cols;
	if(row<7||col<7){
		cout<<"expected size > 7x7"<<endl;
		return -1;
	}
	feature=src(Rect(col/2-3,row/2-3,7,7)).clone();
	imshow("feature",feature);	
	return 0;
}

int histogram(Mat &src,vector<float> &feature){ //R&G channels are used to prodece the histogram
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

int multiHist(Mat &src,vector<float> &feature){ //4 corners and center are the ROI I chosen to concatenate and generate the histogram
	int row=src.rows, col=src.cols;
	Mat tmp=src(Rect(col/2-10,row/2-10,21,21)).clone();
	vconcat(tmp,src(Rect(0,row-21,21,21)).clone(),tmp);
	vconcat(tmp,src(Rect(col-21,0,21,21)).clone(),tmp);
	vconcat(tmp,src(Rect(col-21,row-21,21,21)).clone(),tmp);
	vconcat(tmp,src(Rect(0,0,21,21)).clone(),tmp);
	histogram(tmp,feature);
	return 0;
}

int texture(Mat &src,vector<float> &feature){  //reture a flatten histogram, which is produced by orientation and magnitude of sobel filtered src
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
				//calculate the angle(oritation) of given position
				//and convert it from [-180,180] to [0,360]
				aPtr[tmp]=180+atan2(yPtr[tmp],xPtr[tmp])*180/CV_PI;
				mPtr[tmp]=sqrt(yPtr[tmp]*yPtr[tmp]+xPtr[tmp]*xPtr[tmp]);
				if(min>mPtr[tmp]) min=mPtr[tmp];
				if(max<mPtr[tmp]) max=mPtr[tmp];
			}
		}
	}
	//normalize if are not all zeros
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
int gaborHist(Mat &src,vector<float> &feature){  //calculate the gabor filtered src and then calculate histogram based on that; gabor filter used the function provided by openCv
	Mat gaborKernel = getGaborKernel(cv::Size(30, 30), 1, 0, 1, 0.02);
	Mat tmp;
	src.convertTo(tmp,CV_32F);
	cvtColor(tmp,tmp,COLOR_BGR2GRAY);
	filter2D(tmp,tmp,CV_32F,gaborKernel);
	
	double xmin[4],xmax[4];
    minMaxIdx(tmp,xmin,xmax);

    tmp.convertTo(tmp, CV_8U, 255.0/(xmax[0]-xmin[0]),-255*xmin[0] /(xmax[0] -xmin[0]));
	
	int row=tmp.rows,col=tmp.cols;
	tmp.convertTo(tmp,CV_8U);
	feature=vector<float>(33,0);
	for(int i=0;i<row;i++){
		uchar* cptr=tmp.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			feature[int(cptr[j]/8)]++;
		}
	}
	return 0;
}

//given the directory of images(imgsName), and the name of target image(tarName), return the DNN embedding(feature) of target image
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

//do DFT and then retrieve histogram based on several ROIs of the frequency domain spectrum
int fourierHist(Mat &src,vector<float> &feature){
	int SCALE=8; //this determine the size of every ROI
	float TIME=1/4;  //this determine how much outer part will be abandoned, TIME * 2 = 1/2 in this case
	Mat oriImg;
	cvtColor(src,oriImg, COLOR_BGR2GRAY);

	int row=getOptimalDFTSize(oriImg.rows),col=getOptimalDFTSize(oriImg.cols);
	copyMakeBorder(oriImg,oriImg,0,row-src.rows,0,col-src.cols,BORDER_CONSTANT,Scalar::all(0));
	//oriImg=oriImg(Rect(0,0,col,row));
	Mat planes[]={Mat_<float>(oriImg),Mat::zeros(oriImg.size(),CV_32F)};
	//planes[0].convertTo(planes[0],CV_32F);
	Mat cmplxPlane;
	merge(planes,2,cmplxPlane);
	dft(cmplxPlane,cmplxPlane);
	split(cmplxPlane,planes);
	magnitude(planes[0],planes[1],planes[0]);
	//planes[0]+=Scalar::all(1);
	log(planes[0]+1,planes[0]);
	normalize(planes[0],planes[0],0,1,NORM_MINMAX);
	
	/*upon part is DFT*/
	
	//abandon the outer 1/2 part and then retrive the 4 corners
	Mat featureMat=planes[0](Rect(col*TIME,row*TIME,SCALE,SCALE));
	hconcat(featureMat,planes[0](Rect(col*TIME,row*(1-TIME)-SCALE,SCALE,SCALE)),featureMat);
	hconcat(featureMat,planes[0](Rect(col*(1-TIME)-SCALE,row*TIME,SCALE,SCALE)),featureMat);
	hconcat(featureMat,planes[0](Rect(col*(1-TIME)-SCALE,row*(1-TIME)-SCALE,SCALE,SCALE)),featureMat);
	float x,y,sum;
	if(featureMat.type()!=CV_32F) featureMat.convertTo(src,CV_32F);
	
	feature.assign((float*)featureMat.datastart,(float*)featureMat.dataend);
	//cout<<featureMat.size()<<" "<<feature.size()<<endl;
	for(auto &i:feature) i*=11.843754;
	return 0;
	//11.843754 is the maximum of DNN embedding, this makesure the two part are in same range, with same scale
}
