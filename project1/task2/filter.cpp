/*
Author: Taiwei Cui
TIme: 2024-1-22
*/
#include "filter.h"

//this function implement the function to change an RGB picture to gray scale
//it will use 255 minus the average of three channel as the intensity of gray scale
int myRGB2Gray(const cv::Mat& src,cv::Mat &ans){
	std::vector<cv::Mat> channels(3);
	cv::split(src,channels);
	int row=src.rows, col=src.cols;
	ans=cv::Mat(row,col,CV_8UC1);
	for(int i=0;i<row;i++){
		const uchar* rPtr=channels[0].ptr<uchar>(i);
		const uchar* gPtr=channels[1].ptr<uchar>(i);
		const uchar* bPtr=channels[2].ptr<uchar>(i);
		uchar* ansPtr=ans.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			ansPtr[j]=(uchar)(255-(rPtr[j]+gPtr[j]+bPtr[j])/3.000);
		}
	}
	ans=channels[0];
	return 0;
}

//this function implements the sepia filter
//it take a matrix and split it into 3 channels, then store the result of calculation in new Mat ans
//the edge is darked
int sepia(cv::Mat& src, cv::Mat &ans){
	std::cout<<"her1";
	double sepiaMat[3][3]={{0.272,0.534,0.131},
				{0.349,0.686,0.168},
				{0.393,0.769,0.189}
				};
	std::vector<cv::Mat> channels;
	cv::split(src,channels);
	int row=src.rows, col=src.cols,max=0;
	int ox=col/2,oy=row/2; //this is the center point of the matrix, which is used to dark the edge
	
	ans=src.clone();
	std::cout<<"hrer";
	for(int i=0;i<row;i++){
		const uchar* ori[3]={channels[0].ptr<uchar>(i),
					channels[1].ptr<uchar>(i),
					channels[2].ptr<uchar>(i)
					};
		uchar* ansPtr=ans.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			float posx=j/(float)col,posy=i/(float)row;
			for(int k=0;k<3;k++){
				ansPtr[3*j+k]=0;
				for(int l=0;l<3;l++){
					ansPtr[3*j+k]+=ori[l][j]*sepiaMat[k][l];
				}
				ansPtr[3*j+k]=cv::saturate_cast<uchar>(ori[0][j]*sepiaMat[k][0]+ori[1][j]*sepiaMat[k][1]+ori[2][j]*sepiaMat[k][2]);
				//if(posx*posx+posy+posy>0.25)ansPtr[3*j+k]*=1/(1+posx*posx+posy*posy)
				
				ansPtr[3*j+k]*=1-((2*posx-1)*(2*posx-1)*(2*posx-1)*(2*posx-1)+(2*posy-1)*(2*posy-1)*(2*posy-1)*(2*posy-1))/2;//the further a pixel is from the center, the darker it will be

			}
		}
	}
	
	return 0;

}

//pixel wise blur, which traverses every 5x5 pixels and calculate their Caussian blur
int blur5x5_1(cv::Mat &src, cv::Mat &dst){
	double Gaussian[5][5]={{1,2,4,2,1},
				{2,4,8,4,2},
				{4,8,16,8,4},
				{2,4,8,4,2},
				{1,2,4,2,1}};
	dst=src.clone();
	int col=src.cols, row=src.rows;
	for(int i=2;i<row-2;i++){
		for(int j=2;j<col-2;j++){
			
			dst.at<cv::Vec3b>(i,j)={0,0,0};
			for(int k=0;k<3;k++){
				for(int l=0;l<5;l++){
					for(int v=0;v<5;v++){
						dst.at<cv::Vec3b>(i,j)[k]+=Gaussian[l][v]*src.at<cv::Vec3b>(i+2-l,j+2-v)[k]/100.0;
					}
				}
			}
		}
	}
	return 0;
}


//auxillary calculating function, which helps calculate 1d filter for seperabel filter
//filter is a array of double, n is the size of filter
//specificly designed for tasks in cs5330 project1
//src should be CV_8UC3
int auxCalFunc(double *filter, cv::Mat src, cv::Mat &dst, int n,int sum=-1){
	int col=src.cols,m=n/2, row=src.rows;
	if(sum==-1){
		sum=0;
		for(int i=0;i<n;i++) sum+=filter[i];
		if(sum<=0.00001&&sum>=-0.00001) sum=1;
	}
	dst=src.clone();
	for(int i=0;i<row;i++){
		const uchar* srcPtr=src.ptr<uchar>(i);
		uchar* dstPtr=dst.ptr<uchar>(i);
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

//auxillary calculating function, which helps calculate 1d filter for seperable filter
//filter is a array of double, n is the size of filter
//specificly designed for tasks in cs5330 project1
//src should be CV_16U

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

//this version of blur use seperable filter, which has been implemented above
int blur5x5_2(cv::Mat &src,cv::Mat &dst){
	double Gaussian[5]={1,2,4,2,1};
	/*int row=src.rows,col=src.cols;
	cv::Mat mid=src.clone().t();
	src=src.t();
	for(int i=0;i<col;i++){
		const uchar* srcPtr=src.ptr<uchar>(i);
		uchar* midPtr=mid.ptr<uchar>(i);
		for(int j=2;j<row-2;j++){
			for(int k=0;k<3;k++){
				midPtr[3*j+k]=0;
				for(int l=0;l<5;l++){
					midPtr[3*j+k]+=srcPtr[3*(j-2+l)+k]*Gaussian[l]/10.0;
				}
			}
		}
	}*/
	cv::Mat mid;
	//src=src.t();
	auxCalFunc(Gaussian,src.t(),mid,5); //calculate the verticle filter first;
	//src=src.t();
	/*mid=mid.t();
	src=src.t();
	dst=mid.clone();
	for(int i=2;i<row-2;i++){
		const uchar* midPtr=mid.ptr<uchar>(i);
		uchar* dstPtr=dst.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			for(int k=0;k<3;k++){
				dstPtr[3*j+k]=0;
				for(int l=0;l<5;l++){
					dstPtr[3*j+k]+=midPtr[3*(j-2+l)+k]*Gaussian[l]/10.0;
				}
			}
		}
	}*/
	//mid=mid.t();
	auxCalFunc(Gaussian,mid.t(),dst,5);  //calculate horizonal filter;
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

//blur the image and then quantize, default value of levels is 10, size of blur is 5x5;
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels){
	blur5x5_2(src,dst);
	for(int i=0;i<dst.rows;i++){
		uchar* dstPtr=dst.ptr<uchar>(i);
		for(int j=0;j<dst.cols;j++){
			for(int k=0;k<3;k++){
				dstPtr[3*j+k]-=dstPtr[3*j+k]%levels;
			}
		}
	}

	return 0;
}

// below is for task 11

//this function will enlarge the frame and put the largest face in the center, inspired by ipad
//this function is what used the face detect
int centralize(cv::Mat &tar, cv::Rect& face){
	if(face.width>=tar.cols/2||face.height>=tar.rows/2) return 0;	
	face.x-=face.width/2;
	if(face.x<0) face.x= 0;
	face.y-=face.height/2;
	if(face.y<0) face.y= 0;
	face.width*=2;
	if(face.x+face.width>tar.cols) face.x=tar.cols-face.width-1;
	face.height*=2;
	if(face.y+face.height>tar.rows) face.y=tar.rows-face.height-1;
	tar=tar(face);
	resize(tar,tar,cv::Size(640,480));
	return 0;
}



//this function is to versualize a view that can not tell red and green apart
//in Mat dst, blue channel is the average of red and green channels in Mat src, and red and green channels are both the value of blue channels in Mat src
//this is the function that use pixel wise modification;
int confuseRedGreen(cv::Mat &src, cv::Mat &dst){
	int col=src.cols, row=src.rows,tmp=0;
	dst=src.clone();
	for(int i=0;i<row;i++){
		cv::Vec3b* dstPtr=dst.ptr<cv::Vec3b>(i);
		for(int j=0;j<col;j++){
			/*if(dstPtr[j][2]>dstPtr[j][1]) dstPtr[j][1]=dstPtr[j][2];
			dstPtr[j][2]=0;*/
			tmp=dstPtr[j][0];
			dstPtr[j][0]=( dstPtr[j][1]+dstPtr[j][2])/2;
			dstPtr[j][1]=tmp;
			dstPtr[j][2]=tmp;
		}
	}
	return 0;
}	


//this function implements a max filter, which means every pixel is replaced by the maximum in its 11x11 neighbors;
//the out skirt is just the copy of Mat src;
//also in a manner of seperable filter to simplify calculation
int localMax(cv::Mat &src, cv::Mat &dst){
	int col=src.cols, row=src.rows,max=0;
	std::vector<cv::Mat> channels;
	cv::split(src,channels);
	for(int x=0;x<channels.size();x++){
	channels[x]=channels[x].t();
	cv::Mat mid=channels[x].clone();	
	for(int i=0;i<col;i++){
		uchar* midPtr=mid.ptr<uchar>(i);
		uchar* srcPtr=channels[x].ptr<uchar>(i);
		for(int j=5;j<row-5;j++){
			max=0;
			for(int k=j-5;k<=j+5;k++){
				if(srcPtr[k]>max) max=srcPtr[k];
			}
			midPtr[j]=max;
		}
	}
	mid=mid.t();
	channels[x]=mid.clone();
	for(int i=0;i<row;i++){
		uchar* dstPtr=channels[x].ptr<uchar>(i);
		uchar* midPtr=mid.ptr<uchar>(i);
		for(int j=5;j<col-5;j++){
			max=0;
			for(int k=j-5;k<=j+5;k++){
				if(midPtr[k]>max) max=midPtr[k];
			}
			dstPtr[j]=max;
		}	

	}
	}
	cv::merge(channels,dst);
	return 0;
}
		

//this function can change the contrast and brightness of picture
//this function take 2 extra parameter;
//alpha is the scaler for contrast and beta is scaler for brightness
int brightAndContrast(cv::Mat &src, cv::Mat &dst,float& alpha,int& beta){
	std::vector<cv::Mat> channels;
	cv::split(src,channels);
	for(int i=0;i<channels.size();i++){
		channels[i]=channels[i]*alpha+beta;
	}
	cv::merge(channels,dst);
	return 0;
}
	

//this function implements a kaleidoscope
//it divided the image into several districts, and every district will be exchanged with copy and rotation of its left top part
//it is nothing similar with the kaleidoscope that I am familiar in my childhood, however
int kaleidoscope(cv::Mat &src, cv::Mat &dst,cv::Mat& mask,int& num,int& size){
	int col=src.cols;

	if(col!=src.rows) {
		std::cout<<"expecting square"<<std::endl;
		return 0;
	}
	std::vector<cv::Mat> channels;
	cv::split(src,channels);
	cv::Mat core,coreN;

	for(int i=0;i<num;i++){
		for(int j=0;j<num;j++){
			for(int z=0;z<3;z++){
				core=channels[z](cv::Rect(i*size,j*size,size,size)); 
				//every core is a district to filter, this mask will copy and rotate the left top part of every core
				//I will show more detail in report
				core.convertTo(core,CV_32F);
				/*for(int x=0;x<size/2;x++){
					float* corePtr=core.ptr<float>(x);
					float* corePtr2=core.ptr<float>(x+(size+1)/2);
					for(int y=0;y<size/2;y++){
						corePtr[y]=0;	
						corePtr2[y]=0;
						corePtr2[y+(size+1)/2]=0;
					}
				}
				*/
				//coreN=mask*core*mask;
				cv::gemm(mask,core,1,cv::Mat(),0,coreN);
				cv::gemm(coreN,mask.t(),1,cv::Mat(),0,coreN);
				coreN.convertTo(coreN,CV_8U);
				coreN.copyTo(channels[z](cv::Rect(i*size,j*size,size,size)));
			}
		}
	}

	cv::merge(channels,dst);
	return 0;

}
