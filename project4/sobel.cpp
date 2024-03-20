#include "sobel.h"
int auxCalFunc_16U(double *filter, cv::Mat src, cv::Mat &dst, int n,int sum=-1){

        int col=src.cols,m=n/2, row=src.rows;
        if(sum==-1){
                sum=0;
                for(int i=0;i<n;i++) sum+=filter[i];
                if(sum<=0.00001&&sum>=-0.00001) sum=1;
        }
        dst=src.clone();
        for(int i=0;i<row;i++){
                const short* srcPtr=src.ptr<short>(i);
                short* dstPtr=dst.ptr<short>(i);
                for(int j=m;j<col-m;j++){
                        for(int k=0;k<3;k++){
                                dstPtr[3*j+k]=0;
                                for(int l=0;l<n;l++){
                                        dstPtr[3*j+k]+=srcPtr[3*(j-m+l)+k]*filter[l]/sum;
                                }
                        }
                }
        }
        return 0;

}
//the following two function implement sobel x and sobel y, both use seperable filter implemented above
int sobelX3x3(cv::Mat &src, cv::Mat &dst){	
// 
//-1 0 1   1
//-2 0 2 = 2 x -1 0 1
//-1 0 1   1
//
	double x1[3]={1.0,2.0,1.0},x2[3]={-1.0,0,1.0};
	cv::Mat mid;
	cv::Mat tmp;
	src.convertTo(tmp,CV_16SC3);
	auxCalFunc_16U(x1,tmp.t(),mid,3,1); //calculate verticle filter
	auxCalFunc_16U(x2,mid.t(),dst,3,1); //calculate horizonal filter
	return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst){
//
//1   2  1   1
//0   0  0 = 0 x 1 2 1
//-1 -2 -1  -1
//
	double y1[3]={1.0,0,-1.0},y2[3]={1.0,2.0,1.0};
	cv::Mat mid,tmp;
	src.convertTo(tmp,CV_16SC3);
	auxCalFunc_16U(y1,tmp.t(),mid,3,1); //calculate verticle filter
	auxCalFunc_16U(y2,mid.t(),dst,3,1); //calculate horizonal filter
	/*
	cv::Mat kernel = (cv::Mat_<float>(1,3) << 1.0, 0, -1.0);
	cv::Mat kernel2=(cv::Mat_<float>(1,3) << 1.0, 2.0, 1.0);
	cv::Mat mid;
	cv::filter2D(src.t(),mid,-1,kernel);
	cv::filter2D(mid.t(),dst,-1,kernel2);
	*/
	return 0;
}

//this function calculate magnitude of image based on sobelx and sobel y'
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
	int col=sx.cols,row=sy.rows;
	dst=cv::Mat(row,col,CV_8UC3);
	
	for(int i=0;i<row;i++){
		const short* sxPtr=sx.ptr<short>(i);
		const short* syPtr=sy.ptr<short>(i);
		uchar* dstPtr=dst.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			for(int k=0;k<3;k++){
				dstPtr[3*j+k]=(uchar)(sqrt(sxPtr[3*j+k]*sxPtr[3*j+k]+syPtr[3*j+k]*syPtr[3*j+k]));//calculate the square root of elements in sobel x and sobel y
			}
		}
	}
	return 0;
}
int integralImg(cv::Mat &src,cv::Mat &itgImg){
	itgImg=cv::Mat(src.size(),src.type());
	int row=src.rows,col=src.cols;
	uchar* cur,* pre=src.ptr<uchar>(0);
	uchar* cur_,* pre_=itgImg.ptr<uchar>(0);
	pre_[0]=pre[0];
	for(int i=1;i<col;i++){
		pre_[i]=pre_[i-1]+pre[i];
	}
	for(int i=1;i<row;i++){
		cur=src.ptr<uchar>(0);
		cur_=itgImg.ptr<uchar>(0);
		cur_[0]=pre_[0]+cur[0];
		for(int j=1;j<col;j++){
			cur_[j]=cur[j]+pre_[j]+cur_[j-1]-pre_[j-1];
		}
		
	}

}
