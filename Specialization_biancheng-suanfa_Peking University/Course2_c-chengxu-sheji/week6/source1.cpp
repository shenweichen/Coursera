#include <iostream>
#include <algorithm>
#include <iomanip>
using namespace std;
struct P{
	int id;
	double value;
}p[50];

bool cmp(P a,P b){
	return a.value>b.value;
}

int main(){
	int m;
	double a;
	cin>>m>>a;
	for(int i=0;i<m;i++){
		cin>>p[i].id>>p[i].value;
	}
	sort(p,p+m,cmp);
	int i = 0;
	for(;i<m&&p[i].value>a;i++){
		cout<<setfill('0')<<setw(3)<<p[i].id<<" ";
		cout<<fixed<<setprecision(1)<<p[i].value<<endl;
	}
	if(i==0)
		cout<<"None."<<endl;
}