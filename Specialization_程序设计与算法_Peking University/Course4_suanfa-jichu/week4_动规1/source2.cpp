#include <iostream>
#include <algorithm>
using namespace std;
const int maxh = 10001;
int h[102][102];
int maxlen[102][102]={0};
int findpath(int i,int j){
	if (maxlen[i][j]!=0)//如果计算过就不再计算
		return maxlen[i][j];
	int left=1,right=1,up=1,down=1;
	if(h[i-1][j]<h[i][j]){
		maxlen[i-1][j]=findpath(i-1,j);//计算过程中保存结果
		left = maxlen[i-1][j]+1;
	}
	if(h[i+1][j]<h[i][j]){
		maxlen[i+1][j]=findpath(i+1,j);
		right = maxlen[i+1][j]+1;
	}
	if(h[i][j+1]<h[i][j]){
		maxlen[i][j+1] = findpath(i,j+1);
		up = maxlen[i][j+1]+1;
	}
	if(h[i][j-1]<h[i][j]){
		maxlen[i][j-1]= findpath(i,j-1);
		down = maxlen[i][j-1]+1;
	}
	return max(max(left,right),max(up,down));
}
int main(){
	int row,col,temp;
	cin>>row>>col;
	fill(h[0],h[0]+col+2,maxh);
	fill(h[row+1],h[row+1]+col+2,maxh);
	for(int i=1;i<row+1;i++){
		for(int j=0;j<col+2;j++){
			if(j==0||j==col+1)
				h[i][j]=maxh;
			else{
				cin>>temp;
				h[i][j]=temp;
			}
		}
	}
	temp = 0;
	for(int i=1;i<row+1;i++){
		for(int j=1;j<col+1;j++){
			maxlen[i][j]=findpath(i,j);
			temp = max(temp,maxlen[i][j]);
		}
	}
	cout<<temp<<endl;
}

