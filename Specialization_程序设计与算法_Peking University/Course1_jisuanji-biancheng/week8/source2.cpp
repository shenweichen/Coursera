#include <iostream>
using namespace std;
int main(){
	int n;
	cin>>n;
	while(n--){
		double distance;
		cin>>distance;
		double bike = distance/3+50;
		double walk = distance/1.2;
		if(bike<walk)
			cout<<"Bike"<<endl;
		else if(bike==walk)
			cout<<"All"<<endl;
		else
			cout<<"Walk"<<endl;
	}
}