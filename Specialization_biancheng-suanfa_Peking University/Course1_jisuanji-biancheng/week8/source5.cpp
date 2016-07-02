#include <iostream>
using namespace std;
int main(){
	int n;
	cin>>n;
	for(int i=10;i<=n;i++){
	int temp = i;
	int sum = 0;
	while(temp!=0){
		sum+=temp%10;
		temp/=10;
	}
	if(i%sum==0)
		cout<<i<<endl;
}

}