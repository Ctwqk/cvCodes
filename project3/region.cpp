#include "region.h"
using namespace std;
using namespace cv;
// x: whose root is what we want to find
// array: the union-find set
int find(int x,int**array){
	if(array[x][0]!=x) array[x][0]=find(array[x][0],array);
	return array[x][0];
}
// x&y: given node to union
// array: union-find set
int uni(int x,int y,int **array){
	x=find(x,array);
	y=find(y,array);
	if(x==y){
		return y;
	}
	//if is with same rank, union x to y and increase the rank of y by 1
	if(array[y][1]==array[x][1]){
		array[x][0]=y;
		array[y][1]++;
		return y;
	}
	//else union to the one with higher rank
	else if(array[y][1]>array[x][1]){
		array[x][0]=y;
	}
	else array[y][0]=x;
	//array[x][0]=y;
	//return y;
	return -1;
}
//cut off the border regions with BFS
//array: the flatted image
//row&col: the size of origin image
int cutEdgeRegion2(int *array,int row,int col){
	//EDGE: the range that is regarded as border, any region that connect with border will be deleted.
	//size: the size of the Queue for BFS
	//endPoint: how many point you want to cut off at most
	int EDGE=5,size=4*(row+col),endPoint=row*col/6;
	endPoint=row*col;//I want to erase as much as possible
	int queue[size];
	
	int front=0,rear=0;
	int idx,idx2;
	//push the border into queue
	for(int i=0;i<=EDGE;i++){
		idx=(row-1-i)*col;
		idx2=i*col;
		for(int j=0;j<col;j++){
			if(i==EDGE){
				if(array[idx]>=0) queue[rear++]=idx;
				if(array[idx2]>=0) queue[rear++]=idx2;
			}
			array[idx++]=-2;
			array[idx2++]=-2;
				
		}
		for(int j=EDGE+1;j<row-EDGE-1;j++){
			idx=col*j;
			if(i==EDGE){
				if(array[idx+i]>=0) queue[rear++]=idx+i;
				if(array[idx+col-i-1]>=0) queue[rear++]=idx+col-i-1;
			}
			array[idx+i]=-2;
			array[idx+col-i-1]=-2;
		}
	}

	//4-connected
	int dir[][2]={{0,1},{1,0},{0,-1},{-1,0}/*,{1,-1},{-1,1},{1,1},{-1,-1}*/};

	int z,count=0;
	//BFS
	while(rear!=front&&count<endPoint){
		int tmp=queue[front%size];
		front++;
		idx=tmp/col;
		idx2=tmp%col;
		for(int i=0;i<4;i++){
			z=(idx+dir[i][1])*col+idx2+dir[i][0];
			if(array[z]>=0){
				array[z]=-2;
				count++;
				queue[rear%size]=z;
				rear++;
				if(rear-front>=size){
					cout<<"out of size"<<endl;
					return -1;
				}
			}
		}
		rear%=size;
		front%=size;
		if(front>rear) rear+=size;
	}
	//showArr(array,row,col);
	return 0;
}

//a more naive method to cut off edge
int cutEdgeRegion(int* array,int row,int col){
	int EDGE=10; //define the width of edge;
	int idx,idx2,size=col*row;
	for(int i=0;i<EDGE;i++){
		idx=(row-1-i)*col;
		idx2=i*col;
		for(int j=0;j<col;j++){
			array[idx++]=-2;
			array[idx2++]=-2;
		}
		for(int j=EDGE;j<row-EDGE;j++){
			idx=col*j;
			array[idx+i]=-2;
			array[idx+col-i-1]=-2;
		}
	}
	int m=min(row,col);
	for(int i=EDGE;i<m;i++){
		idx=i*col;
		for(int j=0;j<col-0;j++){
			if(array[idx+j]>=0&&array[idx+j-col]==-2)array[idx+j]=-2;
			if(array[size-idx-col+j]>=0&&array[size-idx+j]==-2)array[size-idx-col+j]=-2;
		}
		for(int j=0;j<row-0;j++){
			idx=j*col;
			if(array[idx+i]>=0&&array[idx+i-1]==-2) array[col+i]=-2;
			if(array[idx+col-i-1]>=0&&array[idx+col-i]==-2) array[idx+col-i-1]=-2;
		}	
	}
	return 0;
}	

//this function depart the given image into regions
//src: given binary image
//dst: output image with each region in different color
//num: maximum number of region, default to 6 
vector<Mat> regionize(Mat &src, Mat &dst,int num){
	//for debug
	/*
	double min,max;
	Point minl,maxl;
	minMaxLoc(src,&min,&max,&minl,&maxl);
	cout<<max;
	*/
	if(src.type()!=CV_8U){
		src.convertTo(src,CV_8U);
	}
	src/=255;
	int row=src.rows, col=src.cols;
	int elemNum=src.total();
	int** array=new int*[elemNum];
	int *array2=new int[elemNum]();
	int tmp=0;
	//union-find need 2-D array, but edge cutting want 1-D array so...
	for(int i=0;i<row;i++){
		uchar* ptr=src.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			tmp=i*col+j;
			array[tmp]=new int[2]();
			//if(!ptr[j]){
			if(ptr[j]==0){
				array2[tmp]=-1;
			}
			else {
				array2[tmp]=tmp;
			}
		}
	}

	cutEdgeRegion2(array2,row,col);
	//union-find to seperate regions
	for(int i=0;i<elemNum;i++)array[i][0]=array2[i];
	int idx;			
	for(int i=0;i<row-1;i++){
		int j=0;
		idx=i*col;
		for(j=0;j<col-1;j++){
			if(array[idx][0]<0){
				idx++;
			    continue;	
			}
			if(array[idx+1][0]>=0) uni(idx,idx+1,array);
			if(array[idx+col][0]>=0) uni(idx,idx+col,array);
			idx++;
		}
		if(array[idx][0]>=0&&array[idx+col][0]>=0){
			uni(idx,idx+col,array);
		}	
	}
	for(int j=0;j<col-1;j++){
		idx++;
		if(array[idx][0]>=0&&array[idx+1][0]>=0) uni(idx,idx+1,array);
	}
	dst=Mat::zeros(src.size(),CV_32F);
	//counting the number of pixels of each region
	map<int,int> check;
	for(int i=0;i<row;i++){
		float *ptr=dst.ptr<float>(i);
		idx=col*i;
		for(int j=0;j<col;j++){
			if(array[idx][0]<0){
				idx++;
				continue;
			}	
			tmp=find(idx,array);
			if(check.find(tmp)==check.end()) check[tmp]=1;
			else check[tmp]++;
			ptr[j]=array[idx][0];
			idx++;
		}
	}
	//sort regions based on their area
	struct comp{
		bool operator()(const pair<int,int> &a,const pair<int,int>&b) const{
			return a.second==b.second?a.first>b.first:a.second>b.second;
		}
	};
	set<pair<int,int> ,comp> ans(check.begin(),check.end());
	Mat *planes=new Mat[6];
	fill(planes,planes+6,Mat::zeros(src.size(),CV_8U));
	idx=0;
	//choose first 6 largest region and ignore the regions less than 1000 pixels
	for(pair<int,int> p:ans){
		if(p.first==0||p.second<1000||idx>=num) break;

		compare(dst,p.first,planes[idx++],CMP_EQ);
	}
	vector<Mat> maps;
	transform(planes,planes+6,back_inserter<vector<Mat>>(maps),[](const Mat &map){return map.clone();});
	//color these regions in order of Blue Green Red Cyan Yellow ane Magenta
	for(int i=0;i<3;i++){
		add(planes[3+i],planes[i],planes[i]);
		add(planes[3+i],planes[(i+1)%3],planes[(i+1)%3]);
	}
	merge(planes,3,dst);
	dst*=255;
	delete[] planes;
	delete[] array;
	delete[] array2;
	return maps;
}
