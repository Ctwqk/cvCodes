#include "region.h"
using namespace std;
using namespace cv;
/*
int find(int x,int**array,int &size){
	if(x>=size){
		cout<<"query out of range: x"<<endl;
		return -1;
	}
	int tmp[4096],idx=0;	
	while(x!=array[x][0]){
		if(array[x][0]<0) {
			cout<<x<<": catch"<<endl;
			break;
		}
		if(idx<4096)tmp[idx++]=x;
		x=array[x][0];	
	}
	for(int i=0;i<idx;i++){
		array[tmp[i]][0]=x;
	}
	return x;
}

*/

int find(int x,int**array){
	if(array[x][0]!=x){
		array[x][0]=find(array[x][0],array);
	}
	return array[x][0];
}
int uni(int x,int y,int **array){
	//x=find(x,array,size);
	//y=find(y,array,size);
	x=find(x,array);
	y=find(y,array);
	if(x==y){
		return y;
	}
	if(array[x][1]==array[y][1]){
		array[x][0]=y;
		array[y][1]++;
		return y;
	}
	else if(array[x][1]>array[y][1]){
		array[y][0]=x;
		return x;
	}
	else {
		array[x][0]=y;
		return y;
	}
	return -1;
}
/*
*/
void showArr(int* array,int row,int col){
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			cout<<array[i*col+j]<<" "; }
		cout<<endl;
	}
	cout<<endl;
}
int cutEdgeRegion2(int *array,int row,int col){
	int EDGE=5,size=4*(row+col),endPoint=row*col/6;
	int queue[size];
	int front=0,rear=0;
	int idx,idx2;
	for(int i=0;i<EDGE;i++){
		idx=(row-1-i)*col;
		idx2=i*col;
		for(int j=0;j<col;j++){
			array[idx++]=-2;
			array[idx2++]=-2;
		}
		for(int j=EDGE+1;j<row-EDGE-1;j++){
			idx=col*j;
			array[idx+i]=-2;
			array[idx+col-i-1]=-2;
		}
	}
	idx=(row-1-EDGE)*col;
	idx2=EDGE*col;
	for(int j=0;j<col;j++){
		if(array[idx]>=0) queue[rear++]=idx;
		if(array[idx2]>=0) queue[rear++]=idx2;
	}
	for(int j=EDGE+1;j<row-EDGE-1;j++){
		idx=col*j;
		if(array[idx+EDGE]>=0) queue[rear++]=idx+EDGE;
		if(array[idx+col-EDGE-1]>=0) queue[rear++]=idx+col-EDGE-1;
	}
	
	int dir[][2]={{0,1},{1,0},{0,-1},{-1,0}/*,{1,-1},{-1,1},{1,1},{-1,-1}*/};
	int z,count=0;
	while(rear!=front&&count<endPoint){
		int n=rear-front;
		for(int b=0;b<n;b++){
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
	}
	//showArr(array,row,col);
	return 0;
}

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
	int m=row>col?col:row;
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
	int *array2=new int[elemNum];
	int tmp=0;
	for(int i=0;i<row;i++){
		uchar* ptr=src.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			tmp=i*col+j;
			//if(!ptr[j]){
			array[tmp]=new int[2]();
			if(ptr[j]==0){
				array2[tmp]=-1;
			}
			else {
				array2[tmp]=tmp;
			}
		}
	}
	for(int i=0;i<elemNum;i++){
		if(array2[i]>=0) array[i][0]=array2[i];
	}
	cutEdgeRegion2(array2,row,col);
	int idx;			
	int cache[2048];
	/*
	auto find =[array,&cache,&idx](int x){
		idx=0;
		while(array[x][0]!=x){
			if(idx<2048)cache[idx++]=x;
			x=array[x][0];
		}
		for(int i=0;i<idx;i++)array[cache[i]][0]=x;
		return x;
	};
	*/
	/*
	auto uni=[array](int x,int y){
		x=find(x,array);
		y=find(y,array);
		if(x==y) return y;
		if(array[x][1]==array[y][1]){
			array[x][0]=y;
			array[y][1]++;
			return y;
		}
		else if(array[x][1]>array[y][1]){
			array[y][0]=x;
			return x;
		}
		else {
			array[x][0]=y;
			return y;
		}
		return -1;
	};
	*/
		
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
			//tmp=find(idx,array,elemNum);
			if(check.find(tmp)==check.end()) check[tmp]=1;
			else check[tmp]++;
			ptr[j]=array[idx][0];
			idx++;
		}
	}
	struct comp{
		bool operator()(const pair<int,int> &a,const pair<int,int>&b) const{
			return a.second==b.second?a.first>b.first:a.second>b.second;
		}
	};

	set<pair<int,int> ,comp> ans(check.begin(),check.end());
	Mat *planes=new Mat[num];
	fill(planes,planes+6,Mat::zeros(src.size(),CV_8U));
	idx=0;
	for(pair<int,int> p:ans){
		if(p.first==0||p.second<1000||idx>=num) break;

		compare(dst,p.first,planes[idx++],CMP_EQ);
	}
	vector<Mat> maps;
	transform(planes,planes+6,back_inserter<vector<Mat>>(maps),[](const Mat &map){return map.clone();});

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
