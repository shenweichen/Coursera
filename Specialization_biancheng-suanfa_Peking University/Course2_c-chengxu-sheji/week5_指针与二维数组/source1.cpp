#include <iostream>
using namespace std;

int main( ){
	int k;
	cin>>k;
	while(k--){
		int m,n;
		int sum = 0;
		cin>>m>>n;
		int a[100][100];
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++)
				cin>>a[i][j];
		}
	if(m==1){
		for(int i=0;i<n;i++)
			sum+=a[0][i];
	}else
	if(n==1){
		for(int i=0;i<m;i++)
			sum+=a[i][0];
	}else{

		for(int i=0;i<n;i++)
			sum+=(a[0][i]+a[m-1][i]);
	
		for(int i=1;i<m-1;i++)
			sum+=(a[i][0]+a[i][n-1]);}
	cout<<sum<<endl;

}
}