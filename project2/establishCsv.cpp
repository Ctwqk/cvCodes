#include "csv_util.h"
#include "features.h"
#include "readfile.h"
#include <fstream>
using namespace std;
using namespace cv;

int main(int argc, char ** argv){
	vector<string > images;
	listImgs(argv[1],images);
	Mat img,feature;
	vector<float> vecFeature,vecFeature2;
	//char s[256];
	int idx=0,flag=0;	
	string targetFeature;
	
	//for given type of feature, set the path toward certain csv file
	if(string(argv[3])=="7x7central") {
		targetFeature=string(argv[2])+"/7x7csv";		
	}
	else if(string(argv[3])=="histogram"){
		targetFeature=string(argv[2])+"/histcsv";
	}
	else if(string(argv[3])=="multihist"){
		targetFeature=string(argv[2])+"/multihistcsv";
	}
	else if(string(argv[3])=="texturecolor"){
		targetFeature=string(argv[2])+"/textcolor";
	}
	else if(string(argv[3])=="gabor"){
		targetFeature=string(argv[2])+"/gaborhist";
	}
	else if(string(argv[3])=="mycustom"){
		targetFeature=string(argv[2])+"/mycustom";
	}

	//if the csv file doesn't exist, create it
	ifstream file(targetFeature);
	if(!file) {
		ofstream newFile(targetFeature);	
		if(newFile.is_open()) newFile.close();
	}
	//if it exists, set reset flag to 1, which means the csv file will be reset
	else flag=1;
	char * targetFeature_=new char[targetFeature.length()+1];
	strcpy(targetFeature_,targetFeature.c_str());

	//for given feature type, calculate features for every image and restore the result in the csv file
	if(string(argv[3])=="7x7central"){
		for (string s:images){
			img=imread(s);
			central7x7(img,feature);
			vector<float> tmp=mat2vec(feature);
			char* swa=new char[s.size()+1];	
			strcpy(swa,s.c_str());
			append_image_data_csv(targetFeature_, swa,tmp,flag);
			if(flag) flag=0;
			delete[] swa;
		}		
	}
	else if(string(argv[3])=="histogram"){
		for (string s:images){
			img=imread(s);
			histogram(img,vecFeature);
			char* swa=new char[s.size()+1];	
			strcpy(swa,s.c_str());
			append_image_data_csv(targetFeature_, swa,vecFeature,flag);
			flag=0;
			delete[] swa;
		}
	}
	else if(string(argv[3])=="multihist"){
		for (string s:images){
			img=imread(s);
			multiHist(img,vecFeature);
			char* swa=new char[s.size()+1];	
			strcpy(swa,s.c_str());
			append_image_data_csv(targetFeature_, swa,vecFeature,flag);
			flag=0;
			delete[] swa;
		}
	}
	else if(string(argv[3])=="texturecolor"){
		for (string s:images){
			img=imread(s);
			texture(img,vecFeature);
			histogram(img,vecFeature2);
			vecFeature.insert(vecFeature.end(),vecFeature2.begin(),vecFeature2.end());
			char* swa=new char[s.size()+1];	
			strcpy(swa,s.c_str());
			append_image_data_csv(targetFeature_, swa,vecFeature,flag);
			flag=0;
			delete[] swa;
			//cout<<"done"<<endl;
		}
	}
	else if(string(argv[3])=="gabor"){
		for (string s:images){
			img=imread(s);
			gaborHist(img,vecFeature);
			char* swa=new char[s.size()+1];	
			strcpy(swa,s.c_str());
			append_image_data_csv(targetFeature_, swa,vecFeature,flag);
			flag=0;
			delete[] swa;
		}
	}

	else if(string(argv[3])=="mycustom"){
		for(string s:images){
			img=imread(s);
			fourierHist(img,vecFeature);
			searchCsv("featureCsvs/ResNet18_olym.csv",s,vecFeature2);		
			//cout<<vecFeature.size()<<" "<<vecFeature2.size()<<endl;
			vecFeature.insert(vecFeature.end(),vecFeature2.begin(),vecFeature2.end());
			char* swa=new char[s.size()+1];	
			strcpy(swa,s.c_str());
			append_image_data_csv(targetFeature_, swa,vecFeature,flag);
			flag=0;
			delete[] swa;
		}
	}
	return 0;
} 
