#include <iostream>
#include <iomanip>
using namespace std;
int main(){
	int number;
	int s = 0;
	cin>>number;
	cout<<hex<<number<<endl;
	cout<<setw(10)<<setfill('0')<<dec<<number<<endl;
}