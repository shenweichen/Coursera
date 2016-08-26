#include <iostream>
#include <iomanip>
using namespace std;
int main(){
	int n;
	cin>>n;
	if(n<60)
		cout<<"7";
	else if(n<70)
		cout<<"6";
	else if(n<80)
		cout<<"5";
	else if(n<85)
		cout<<"4";
	else if(n<90)
		cout<<"3";
	else if(n<95)
		cout<<"2";
	else if(n<=100)
		cout<<"1";
}