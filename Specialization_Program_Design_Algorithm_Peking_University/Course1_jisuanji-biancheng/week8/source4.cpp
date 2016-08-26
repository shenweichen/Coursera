#include <iostream>
#include <algorithm>
using namespace std;
int main(){
	int n,k;
	int a[1000];
	cin >> n >> k;
	for(int i =0;i<n;i++)
		cin>>a[i];
	sort(a,a+n);
	int i=0,j=n-1;
	while(i<j){
		if(a[i]+a[j]>k)
			j--;
		else if(a[i]+a[j]<k)
			i++;
		else{
			cout<<"yes"<<endl;
			break;
		}
	}
	if(i>=j)
		cout<<"no"<<endl;
}