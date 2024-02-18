#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(int argc,char **argv){
	if(argc<2){
		cout<<"require input"<<endl;
		return 0;
	}
	imread(argv[1],IMREAD_GRAYSCALE);
		
    return 0;
}
