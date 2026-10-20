/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * save to or read from given .txt file the patterns of images.
 */
#include "saveOrReadPattern.hpp"
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
		mat_3ds.push_back((Mat_<double> (3,4) << 0,0,x,x,0,y,0,y,1,1,1,1));
		if(x>y){
			x=1;
			y/=x;
		}
		else{
			y=1;
			x/=y;
		}
		points2ds.push_back({Point2f(0,0),Point2f(0,y),Point2f(x,0),Point2f(x,y)});
		points3ds.push_back({Point3f(0,0,0),Point3f(0,y,0),Point3f(x,0,0),Point3f(x,y,0)});

		inStream>>row;
		inStream>>col;
	}
	inStream.close();

	return 0;
}
int establishPatterns(std::string imageFilePath, std::string patternFilePath,int minHessian,int mode){
	vector<Mat> images;
	Ptr<xfeatures2d::SURF> detector=xfeatures2d::SURF::create(minHessian);
	Mat descriptor;
	vector<KeyPoint> keypoints;
	for(const auto& entry: fs::directory_iterator(imageFilePath)){
		if(entry.is_regular_file()){
			string filepath = entry.path().string();
			Mat img = imread(filepath,IMREAD_GRAYSCALE);
			if(!img.empty()){
				detector->detectAndCompute(img,Mat(),keypoints,descriptor);
				//cout<<descriptor.rows<<" "<<descriptor.cols<<endl;;
				writePatternToCsv(keypoints,descriptor,img.rows,img.cols,patternFilePath);
				Mat tmp;
				cv::drawKeypoints(img,keypoints,tmp,Scalar::all(-1));	
				cout<<keypoints.size()<<endl;
				imshow("check",tmp);
				waitKey(0);
				
			}
		}
	}
	return 0;
}
