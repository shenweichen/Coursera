#include <iostream>
using namespace std;
int main(){
	int N,K;
	while(cin>>N>>K){
		int sum = N;
		int year = 1;
		double price = 200.0;
		while(year<=20){
			if(sum>price){
				cout<<year<<endl;
				break;
			}
			price *=(double(K)/100+1);
			sum+=N;
			year++;
		}
		if(year>20)
			cout<<"Impossible"<<endl;

	}
}