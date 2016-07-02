#include <iostream>
#include <cmath>
using namespace std;
int main() {
	int n;
	cin >> n;
	int a[3];
	int size = 0;
	while(n!=0){
		a[size++]=n%10;
		n/=10;
	}
	for(int i=2;i>=0;i--){
		cout<<a[i]<<endl;
	}
}
