#include "dist.h"

float dist(vector<float> &feature1,const vector<float> &feature2){
	float ans=0;
	if(true){
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
	return -ans;
}	
	
float histDist(vector<float> &hist1, const vector<float> &hist2){
	float ans=0;
	int n=hist1.size();
	if(hist2.size()!=n){
		cout<<"expected same size of two features: "<<n<<"!="<<hist2.size()<<endl;
		return -1;
	}
	for(int i=0;i<n;i++){
		hist1[i]>hist2[i]?ans+=hist2[i]:ans+=hist1[i];
	}
	return ans;
}
float cosDist(vector<float> &feature1, const vector<float> &feature2){
	int n=feature1.size();
	if(feature2.size()!=n){
		cout<<"expect same size: "<<n<<"!="<<feature2.size()<<endl;
		return -1;
	}
	float sum=0,sum1=0,sum2=0;
	for(int i=0;i<n;i++){
		sum1+=feature1[i]*feature1[i];
		sum2+=feature2[i]*feature2[i];
		sum+=feature1[i]*feature2[i];
	}	
	return -1+sum/sqrt(sum1)/sqrt(sum2);
	
	
	return 0;
}
float myDist(vector<float> &feature1, const vector<float> &feature2){
	return  cosDist(feature1,feature2);
}
