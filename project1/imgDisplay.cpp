#include "task2/filter.h"
using namespace cv;
using namespace std;
float avgEng(Mat frame){
	int row=frame.rows,col=3*frame.cols;
	if(frame.type()!=CV_32F) frame.convertTo(frame,CV_32F);
	float sum=0;
	for(int i=0;i<row;i++){
		float* ptr=frame.ptr<float>(i);
		float tmp=0;
		for(int j=0;j<col;j++){
			tmp+=ptr[j];
		}
		sum+=tmp/col;
	}
	return sum/row;
}

int main(int argc, char** argv){// argv[1] is the position of img to show
	Mat picture=imread(argv[1],IMREAD_COLOR);

	namedWindow("picture",WINDOW_NORMAL);
	Mat frame=picture;
	for(;;){
		char c=waitKey(10);//waiting for key input
		if(c=='q') break;
		else if(c=='g'){ //show the gray scale image
			cvtColor(picture,frame,COLOR_RGB2GRAY);
		}
		else if(c=='p'){  // show the sepia filter with a darker edge
			sepia(picture,frame);
		}
		else if(c=='j'){
			myRGB2Gray(picture,frame);
		}
		else if(c=='o'){  //oirgin picture
			frame=picture;
		}	
		else if(c=='b'){
			blur5x5_1(picture,frame);	
		}
		else if(c=='l'){
			blurQuantize(picture,frame);
		}
		else if (c=='m'){
			Mat sy,sx;
			sobelX3x3(picture,sx);
			sobelY3x3(picture,sy);
			magnitude(sx,sy,frame);
			cout<<avgEng(frame)<<endl;
		}
		else if(c=='s'){  //save image
			int idx=0;
			string filename="savedImg(0).jpg";
			std::ifstream file(filename);
			while(file.good()){  //save image with the first not exist file name
				filename="savedImg("+std::to_string(++idx)+").jpg";
				file=std::ifstream(filename);
			}
			if(imwrite(filename,frame)){
				printf("saved\n");
			}
		}
		resizeWindow("picture",800,480);
		imshow("picture",frame);	
	}
	return 0;
}
