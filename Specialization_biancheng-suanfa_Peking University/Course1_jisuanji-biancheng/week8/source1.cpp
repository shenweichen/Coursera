#include <iostream>
using namespace std;
int main(){
	int a;
	cin>>a;
	int sum = 0 ;
	for(int i=0;i<5;i++){
		int temp;
		cin>>temp;
		if(temp<a)
			sum+=temp;
	}
	cout<<sum;
}