#include <iostream>
#include <map>
#include <iomanip>
using namespace std;
int main() {
	double sum[3]={0};
	char TYPE[4]="ABC";
	map<char,double> mp;
	for(int line =0;line <3 ;line ++){
		int id,n;
		cin>>id>>n;
		while(n--){
			char type;
			double price;
			cin>>type>>price;
			sum[id-1]+=price;
			mp.count(type)==0?mp[type]=1:mp[type]+=price;
		}
	}
	for(int i=0;i<3;i++)
		cout<<i+1<<" "<<fixed<<setprecision(2)<<sum[i]<<endl;
	for(int i=0;i<3;i++)
		cout<<TYPE[i]<<" "<<fixed<<setprecision(2)<<mp[TYPE[i]]<<endl;

}