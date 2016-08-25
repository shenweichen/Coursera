#include <iostream>
using namespace std;
int main() {
	int n;
	while(cin>>n){
	int tag[8]={0};
	if(n%3==0)
		tag[3] =1;
	if(n%5==0)
		tag[5] =1;
	if(n%7==0)
		tag[7] =1;
	for(int i=0;i<8;i++){
		if(tag[i]){
			cout<<i<<" ";
			tag[0]=1;
		}
	}
	if(!tag[0])
		cout<<"n";
	cout<<endl;
}
}