#include <iostream>
#include <iomanip>
using namespace std;
int main(){
	double number;
	int s = 0;
	cin>>number;
	cout<<fixed<<setprecision(5)<<number<<endl;
	cout.unsetf(ios::fixed);//remove the setf(ios::fixed) 
	cout.setf(ios::scientific);
    cout.precision(7);
	cout<<number<<endl;
}