#include <iostream>
#include <set>
#include <algorithm>
using namespace std;
multiset<int> s;
set<int> hasht;
int main(){
	int n;
	cin>>n;
	while(n--){
		string str;
		cin>>str;
		int num;
		switch(str[1]){
			case 'd':{cin>>num;s.insert(num);hasht.insert(num);cout<<s.count(num)<<endl;break;}
			case 'e':{cin>>num;cout<<s.count(num)<<endl;s.erase(num);break;}
			case 's':{
					cin>>num;
					if(hasht.find(num)!=hasht.end())
					cout<<1<<" ";
					else
						cout<<0<<" ";
					cout<<s.count(num)<<endl;


			}
		}
	}
}