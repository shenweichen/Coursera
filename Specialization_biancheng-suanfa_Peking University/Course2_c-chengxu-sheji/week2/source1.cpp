#include <iostream>
using namespace std;
int main() {
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		int temp;
		cin>>temp;
		if(temp==i){
			cout<<i;
			return 0;
		}
	}
	cout<<"N";
	return 0;
}