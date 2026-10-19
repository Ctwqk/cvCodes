/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * save to or read from given .txt file the patterns of images.
 */
#include "saveOrReadPattern.hpp"
#include "app_io.h"
#include <algorithm>
#include <iomanip>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int writePatternToCsv(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor, int& x, int& y,string filepath,int mode){//0 is appending mode;
	ofstream outStream;
	if(mode){
		outStream.open(filepath);
	}
	else{
		outStream.open(filepath,ios::app);
	}

	if(!outStream){
		cout<<"failed to open or create given file"<<endl;
		return -1;
	}
	if (descriptor.type() != CV_32F || descriptor.rows != static_cast<int>(keypoints.size())) return -1;
	outStream << std::setprecision(9);
	int row=descriptor.rows, col=descriptor.cols;
	outStream<<row<<" ";
	outStream<<col<<" ";
	outStream<<keypoints.size()<<" ";
	for(KeyPoint k : keypoints){
		outStream<<k.pt.x<<" ";
		outStream<<k.pt.y<<" ";
		outStream<<k.size<<" ";
		outStream<<k.angle<<" ";
		outStream<<k.response<<" ";
		outStream<<k.octave<<" ";
	}
	for(int i=0;i<row;i++){
		const float* ptr=descriptor.ptr<float>(i);
		for(int j=0;j<col;j++){
			outStream<<ptr[j]<<" ";
		}
	}
	outStream<<x<<" ";
	outStream<<y<<endl;
	outStream.close();
	return 0;
}
int readPatternsFromCsv(string csvFile,std::vector<std::vector<cv::KeyPoint>> &keypoints, std::vector<cv::Mat>&descriptors,vector<Mat> & mat_3ds,vector<vector<Point3f>> &points3ds){
	ifstream inStream(csvFile);
	if(!inStream){
		cout<<"failed to open given file"<<endl;
		return -1;
	}
	descriptors=vector<cv::Mat>();
	keypoints=vector<vector<KeyPoint>>();
	mat_3ds=vector<Mat>();
//	points2ds=vector<vector<Point2f>>();
	vector<vector<Point2f>> points2ds;
	points3ds=vector<vector<Point3f>>();
	int row,col,num;
	int x,y;
	inStream>>row;
	inStream>>col;
	//cout<<row<<" "<<col<<" "<<num<<endl;
	while(inStream>>num){
        if (row <= 0 || row > 100000 || col != 128 || num != row) return -1;
		//cout<<"readed"<<endl;
		keypoints.push_back(vector<KeyPoint>());
		for(int i=0;i<num;i++){
			KeyPoint k;
			inStream>>k.pt.x;
			inStream>>k.pt.y;
			inStream>>k.size;
			inStream>>k.angle;
			inStream>>k.response;
			inStream>>k.octave;
			keypoints.back().push_back(k);
		}
		Mat descriptor=(Mat_<float>(row,col));
		for(int i=0;i<row;i++){
			float* ptr=descriptor.ptr<float>(i);
			for(int j=0;j<col;j++){
				inStream>>ptr[j];
			}
		}
		descriptors.push_back(descriptor);
		//cout<<descriptors.size()<<endl;
		inStream>>x;
		inStream>>y;
        if (!inStream || x <= 0 || y <= 0) return -1;
        // x is image height, y is width. Keep pixel coordinates for projection.
        mat_3ds.push_back((Mat_<double>(3,4) << 0,0,y,y,0,x,0,x,1,1,1,1));
        const float scale = static_cast<float>(std::max(x, y));
        points3ds.push_back({Point3f(0,0,0),Point3f(0,x/scale,0),
                            Point3f(y/scale,0,0),Point3f(y/scale,x/scale,0)});

		inStream>>row;
		inStream>>col;
	}
	inStream.close();

	return descriptors.empty() ? -1 : 0;
}
int establishPatterns(std::string imageFilePath, std::string patternFilePath,int minHessian,int mode){
    const auto detector = SIFT::create(minHessian);
    int written = 0;
    for (const auto& path : cvc::listImages(imageFilePath)) {
        Mat img = imread(path, IMREAD_GRAYSCALE), descriptor;
        vector<KeyPoint> keypoints;
        if (img.empty()) continue;
        detector->detectAndCompute(img, noArray(), keypoints, descriptor);
        if (descriptor.empty()) continue;
        if (writePatternToCsv(keypoints, descriptor, img.rows, img.cols, patternFilePath,
                              written == 0 ? mode : 0)) return -1;
        cout << path << ": " << keypoints.size() << " SIFT features\n";
        ++written;
    }
    return written ? 0 : -1;
}
