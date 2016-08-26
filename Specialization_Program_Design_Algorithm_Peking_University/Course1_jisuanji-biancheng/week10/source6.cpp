#include <iostream>
#include <algorithm>
using namespace std;
int main(){
	int n,m;
	int a[1000];
	cin>>n>>m;
	for(int i=0;i<n;i++)
		cin>>a[i];
	reverse(a,a+n);
	reverse(a,a+m);
	reverse(a+m,a+n);

		for(int i=0;i<n;i++)
		cout<<a[i]<<" ";
	
}