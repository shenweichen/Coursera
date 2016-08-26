#include <iostream>
using namespace std;
int main(){
	int n;
	cin>>n;
	int num[6]={0};
		if(n/100!=0){
			num[0]=n/100;
			n-=100*num[0];
		} if(n/50!=0){
			num[1]=n/50;
			n-=50*num[1];
		} if(n/20!=0){
			num[2]=n/20;
			n-=20*num[2];
		} if(n/10!=0){
			num[3]=n/10;
			n-=10*num[3];
		} if(n/5!=0){
			num[4]=n/5;
			n-=5*num[4];
		} if(n!=0){
			num[5]=n;
		}
	for(int i=0;i<6;i++){
		cout<<num[i]<<endl;
	}
}