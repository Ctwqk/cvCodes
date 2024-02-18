#include "region.h"
using namespace std;
using namespace cv;

int find(int x,int*array,int &size){
	
	if(x>=size){
		cout<<"query out of range: x"<<endl;
		return -1;
	}
	int tmp[256],idx=0;	
	while(x!=array[x]){
		if(array[x]<0) {
			cout<<x<<": catch"<<endl;
			break;
		}
		if(idx<256)tmp[idx++]=x;
		x=array[x];	
	}
	for(int i=0;i<idx;i++){
		array[tmp[i]]=x;
	}
	return x;
}
int uni(int x,int y,int *array,int &size){
	if(x>=size||y>=size){
		cout<<"query out of range: x,y"<<endl;
		return -1;
	}
	x=find(x,array,size);
	y=find(y,array,size);
	if(x==y){
		return y;
	}
	array[x]=y;
	return y;
}
void showArr(int* array,int row,int col){
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			cout<<array[i*col+j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}
int cutEdgeRegion2(int *array,int row,int col){
	int EDGE=5,size=4*(row+col);
	int queue[size];
	
	int front=0,rear=0;
	int idx,idx2;
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
	int dir[][2]={{0,1},{1,0},{0,-1},{-1,0},{1,-1},{-1,1},{1,1},{-1,-1}};

	int x,y,z;
	while(rear!=front){
		int tmp=queue[front%size];
		front++;
		idx=tmp/col;
		idx2=tmp%col;
		for(int i=0;i<4;i++){
			y=idx+dir[i][1];
			x=idx2+dir[i][0];
			z=y*col+x;
			if(array[z]>=0){
				array[z]=-2;
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
int regionize(Mat &src, Mat &dst,int num){
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
	int* array=new int[elemNum]();
	int tmp=0;
	for(int i=0;i<row;i++){
		uchar* ptr=src.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			tmp=i*col+j;
			//if(!ptr[j]){
			if(ptr[j]==0){
				array[tmp]=-1;
			}
			else {
				array[tmp]=tmp;
			}
		}
	}

	if(num>=0) cutEdgeRegion2(array,row,col);
	int idx;			
	for(int i=0;i<row-1;i++){
		int j=0;
		idx=i*col;
		for(j=0;j<col-1;j++){
			if(array[idx]<0){
			//if(prev[j]==0){
				idx++;
			    continue;	
			}
			if(array[idx+1]>=0) uni(idx,idx+1,array,elemNum);
			if(array[idx+col]>=0) uni(idx,idx+col,array,elemNum);
			idx++;
		}
		if(array[idx]>=0&&array[idx+col]>=0){
			uni(idx,idx+col,array,elemNum);
		}	
	}
	for(int j=0;j<col-1;j++){
		idx++;
		if(array[idx]>=0&&array[idx+1]>=0) uni(idx,idx+1,array,elemNum);
	}
	dst=Mat::zeros(src.size(),src.type());

	for(int i=0;i<row;i++){
		uchar* ptr=dst.ptr<uchar>(i);
		tmp=i*col;
		for(int j=0;j<col;j++){
			if(array[tmp]>=0){
				ptr[j]=find(tmp,array,elemNum);
			}
			tmp++;
		}
	}
	
	normalize(dst,dst,0,63,NORM_MINMAX);	

	delete[] array;
	return 0;
}
int mapToColor(Mat &map,Mat &dst,int num){
	if(num>3){
		cout<<"we have only 3 channel"<<endl;
		return -1;
	}
	int row=map.rows,col=map.cols;
	Mat planes[3];
	fill(planes,planes+3,Mat::zeros(row,col,CV_8U));
	int hist[64]={0};
	for(int i=0;i<row;i++){
		uchar* ptr=map.ptr<uchar>(i);
		for(int j=0;j<col;j++){
			if(ptr[j])hist[(int)ptr[j]]++;

		}
	}
	for(int i=0;i<num;i++){
		int pos=distance(hist,max_element(hist,hist+64));
		if(!pos||hist[pos]<1000) break;
		compare(map,pos,planes[i],CMP_EQ);
		hist[pos]=0;
	}
	merge(planes,3,dst);
	dst*=255;
	return 0;
}
	

