#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "dnnEmbedding.h"
#include "preprocess.h"
using namespace std;
using namespace cv;

int main(int argc,char** argv){
	VideoCapture *cmrCap=new VideoCapture(atoi(argv[1]));
	if(!cmrCap->isOpened()){
		cout<<"failed to open camera: "<<argv[1]<<endl;
		return -1;
	}
	int WIDTH=640,HEIGHT=480;
	cmrCap->set(CAP_PROP_FRAME_WIDTH, WIDTH);
	cmrCap->set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
	cv::dnn::Net net=cv::dnn::readNet("mnist_model.onnx");
	double minVal,maxVal;
	Point minLoc,maxLoc;

	Mat frame,src,embedding;
	Rect targetBox(0,0,WIDTH,HEIGHT);
	vector<string> layerNames=net.getLayerNames();
	for(int i=0;i<layerNames.size();i++){
		cout<<"layer"<<i<<": "<<layerNames[i]<<endl;
	}

	for(;;){
		char c=waitKey(3);
		if(c=='q'){
			destroyAllWindows();
			waitKey(1);
			break;
		}
		*cmrCap>>frame;
		//threshold(src,src,100,255,cv::THRESH_BINARY_INV);
		//cvtColor(frame,src,COLOR_BGR2GRAY);
		toBinary(frame,src);
		getEmbedding(src,embedding,targetBox,net,0);
		//cout<<embedding.size()<<endl;
		minMaxLoc(embedding,&minVal,&maxVal,&minLoc,&maxLoc);
		cout<<maxLoc<<", likelyhood: "<<maxVal<<endl;
		imshow("vid",src);
	}	

	return 0;
}
