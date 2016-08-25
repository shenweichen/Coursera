#include <iostream>
using namespace std;
const int MAXN = 20000;
int main() {
	int a;
	cin>>a;
	if(a%4==0){
		if(a%3200==0||(a%100==0&&a%400!=0))
			cout<<"N";
		else
			cout<<"Y";
	}else
	cout<<"N";
cout<<endl;
}