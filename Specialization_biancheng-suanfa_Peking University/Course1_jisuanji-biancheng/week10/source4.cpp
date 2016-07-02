#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;
int main(){
	int a[5][5];
	for(int i =0;i<5;i++){
		for(int j=0;j<5;j++)
			cin>>a[i][j];
	}
	int n,m;
	cin>>n>>m;
	if(n>=0&&n<5&&m>=0&&m<5){

	for(int i =0;i<5;i++){
		if(i==min(n,m)){
			for(int j=0;j<5;j++)
			cout<<setw(4)<<a[max(n,m)][j];
		}
			else if(i==max(n,m)){
				for(int j=0;j<5;j++)
			cout<<setw(4)<<a[min(n,m)][j];
			}
			else{
		for(int j=0;j<5;j++)
			cout<<setw(4)<<a[i][j];}
		cout<<endl;
	}

	}else
	cout<<"error"<<endl;
}