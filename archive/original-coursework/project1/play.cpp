#include "task2/filter.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv){// argv[1] is the position of img to show
	Mat picture=imread(argv[1],IMREAD_COLOR);
	picture=picture(Rect(100,200,300,300));
	Mat ope1=Mat::eye(300,300,CV_32F);
	vector<Mat> channels;
	split(picture,channels);
	
	for(int i=150;i<300;i++) {
		float* eyePtr=ope1.ptr<float>(i);
		eyePtr[i]=0;
		eyePtr[300-1-i]=1;
	}
	imshow("wtf",ope1*255);
	for(int i=0;i<3;i++){
		channels[i].convertTo(channels[i],CV_32F);
		channels[i]=ope1*channels[i]*ope1.t();
		channels[i].convertTo(channels[i],CV_8U);
	}
	merge(channels,picture);
	
	namedWindow("picture",1);
	Mat frame=picture;
	for(;;){
		char c=waitKey(10);//waiting for key input
		if(c=='q') break;
		
		else if(c=='o'){  //oirgin picture
			frame=picture;
		}	
		else if(c=='k'){
				
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

		imshow("picture",frame);	
	}
	return 0;
}
