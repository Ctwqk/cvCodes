#include "task2/filter.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv){// argv[1] is the position of img to show
	Mat picture=imread(argv[1],IMREAD_COLOR);

	namedWindow("picture",1);
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
