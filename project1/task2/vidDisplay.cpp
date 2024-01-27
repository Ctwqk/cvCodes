/*
Author: Taiwei Cui
Time: 2024-1-22
*/
#include "filter.h"
using namespace std;
      
using namespace cv; 
int main() { VideoCapture *capdev;
        // open the video device
        capdev = new VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        namedWindow("Video", 1); // identifies a window
	Mat frame,tmp,swap;
	vector<Rect> faces;
	Rect face;
	int idx;
	int flag=0; //cache of pressed key
	int brightScaler=0;
	float contrastScaler=1;
        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                	printf("frame is empty\n");
                	break;
                }          
                if(flag==1){//Opencv gray scale, press g
                	cvtColor(frame,frame,COLOR_RGB2GRAY);
        	}
		else if(flag==2){ //my gray scale  press h
			myRGB2Gray(frame,frame);
		}
		else if(flag==3){ //blur the image , press b
			blur5x5_2(frame,frame);
		}
		else if(flag==4){ //calculating sobel x, press x
			sobelX3x3(frame,tmp);  //tmp is what actually the calculation result
			convertScaleAbs(tmp,frame);  //frame is what I try to visualize
			
		}
		else if(flag==5){ //calculating sobel y, press y;
			sobelY3x3(frame,tmp);  //tmp is what actually the calculation result
			convertScaleAbs(tmp,frame);  //frame is what I try to visualize
		}
		else if(flag==6){  //calculating the magnitude of image, press m
			sobelX3x3(frame,tmp);
			sobelY3x3(frame,swap);
			magnitude(swap,tmp,frame);
		}
		else if(flag==7){  //calculating the blurQuantize of image, press l
			blurQuantize(frame,frame);
		}
		else if(flag==8){  //detecting and boxing face, press f
			cvtColor(frame,tmp,COLOR_BGR2GRAY,0);
			detectFaces(tmp,faces);
			drawBoxes(frame,faces);
		} 
		else if(flag==9){  //centralizing largest face, press c;
			cvtColor(frame,tmp,COLOR_BGR2GRAY,0);
			detectFaces(tmp,faces);
				if(faces.size()){
				idx=0;	
				for(int i=1;i<2*faces.size()/3;i++){ //only calculating first 2/3 faces to accelerate calculating, because I believe if a face is largest, it will appear in the first 2/3, if it does not, then it is not much larger than the second one
					if(faces[i].width>faces[idx].width) idx=i;
				}
				centralize(frame,faces[idx]);
			}
		}
		else if(flag==10){  //confuse red and green, press r
			confuseRedGreen(frame,frame);
		}
		else if(flag==11){  //kaleidoscope, press k
			int num=5;
			int row=frame.rows;
			if(row>frame.cols) row=frame.cols;
			int size=row/num;
			Mat mask=Mat::eye(size,size,CV_32F);
			for(int i=size/2;i<size;i++){   //produce the mask for kaleidoscope filter
				float* maskPtr=mask.ptr<float>(i);
				maskPtr[i]=0;
				maskPtr[size-1-i]=1;
			}
			frame=frame(Rect(80,0,row,row));
			kaleidoscope(frame,frame,mask,num,size);
		}
		else if(flag==12){
			localMax(frame,frame);//calculate max pixel in its 11x11 neighbors, press e
		}
		brightAndContrast(frame,frame,contrastScaler,brightScaler); //show the image with the scalers for brightness and contrastScaler
                imshow("Video", frame);
                // see if there is a waiting keystroke
                char key = waitKey(10);
                if( key == 'q') {
                	break;
                }
		else if(key=='s'){
		        
			/*if(imwrite("savedImg.jpg",frame)){
				printf("saved\n");
			}*/
			int idx=0;
			string filename="savedImg(0).jpg";
			std::ifstream file(filename);
			while(file.good()){    //save image with first not exist file name
				filename="savedImg("+std::to_string(++idx)+").jpg";
				file=std::ifstream(filename);
			}
			if(imwrite(filename,frame)){
				printf("saved\n");
			}
		}
                else if(key=='g'){
                	flag=1;
                }
		else if(key=='h'){
			flag=2;
		}
		else if(key=='b'){
			flag=3;
		}
		else if(key=='o'){
			flag=0;
		}
		else if(key=='x'){
			flag=4;
		}
		else if(key=='y'){
			flag=5;
		}
		else if(key=='m'){
			flag=6;
		} 
		else if(key=='l'){
			flag=7;
		}
		else if(key=='f'){
			flag=8;
		}
		else if(key=='c'){
			flag=9;
		}
		else if(key=='r'){
			flag=10;
		}
		else if(key=='k'){
			flag=11;
		}
		else if(key=='e'){
			flag=12;
		}
		else if(key=='+'&&brightScaler<50){   //brighter, no larger then 50
			brightScaler+=5;
		}
		else if(key=='-'&&brightScaler>-50){  //darker, no smaller then -50
			brightScaler-=5;
		}
		else if(key=='>'&&contrastScaler<2){ //higher contrast, no higher then 2
			contrastScaler+=0.05;
		}
		else if(key=='<'&&contrastScaler>0.5){  //lower contrast, no lower then 0.5
			contrastScaler-=0.05;
		}
		else if(key=='a'){   //reset the contrast and brightness
			brightScaler=0;
			contrastScaler=1;
		}
       }

        delete capdev;
        return(0);
}

