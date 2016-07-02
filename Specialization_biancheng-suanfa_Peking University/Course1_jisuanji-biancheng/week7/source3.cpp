#include <iostream>
using namespace std;
int main(){
	 int num[1000]={0};
	 int n,k;
	 cin>>n>>k;
	 for(int i=0;i<n;i++){
	 	int temp;
	 	cin>>temp;
	 	num[temp]++;
	 }
	 int count = 0;
	 for(int i=999;i>=0;i--){
	 	while(num[i]!=0){
	 		num[i]--;
	 		count++;
	 		if(count == k){
	 			cout<<i<<endl;
	 			break;
	 		}
	 	}
	 	if(count==k)
	 		break;
	 }
}