#include <iostream>
using namespace std;
int main(){
	int dp[31]={0};
	int n;
	dp[0]=1;
	dp[2]=3;
	dp[4]=11;
	int sum_i=14;
	for(int i=6;i<31;i+=2){
		dp[i]=dp[i-2]+2*sum_i+2;
		sum_i+=dp[i];
	}
	while(cin>>n){
		if(n==-1)
			break;
		cout<<dp[n]<<endl;

	}
	return 0;
}