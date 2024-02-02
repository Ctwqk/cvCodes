#include <iostream>

using namespace std;

float dist(vector<float> &feature1, vector<float> &feature2,char type[128]){
	float ans=0;
	if(!strcmp(type,"sum-of-squared-difference")){
		int n=feature1.size();
		if(n!=feature2.size()) {
			cout<<"expected same size of two features"<<endl;
			return -1;
		}
		float tmp=0;
		for(int i=0;i<n;i++){
			tmp=feature1[i]-feature2[i];
			ans+=tmp*tmp;
		}
	}
	return ans;


}	
	
