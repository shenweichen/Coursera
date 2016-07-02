#include <iostream>
#include <iomanip>
using namespace std;
int main(){
	int n;
	double num[4]={0.0};
	cin>>n;
	for(int i=0;i<n;i++){
		int age;
		cin >> age;
		if(age<=18)
			num[0]++;
		else if(age<=35)
			num[1]++;
		else if(age<=60)
			num[2]++;
		else
			num[3]++;
	}
	cout<<fixed<<setprecision(2);
	cout<<"1-18: "<<num[0]*100/n<<"%"<<endl;
	cout<<"19-35: "<<num[1]*100/n<<"%"<<endl;
	cout<<"36-60: "<<num[2]*100/n<<"%"<<endl;
	cout<<"60-: "<<num[3]*100/n<<"%"<<endl;
}