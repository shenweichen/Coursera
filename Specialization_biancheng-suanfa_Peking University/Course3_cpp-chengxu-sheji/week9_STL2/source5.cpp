#include <iostream>
#include <map>
using namespace std;
typedef map<int,int>  Member;
Member member;
int main(){
	int n;
	member[1000000000]=1;
	cin>>n;
	while(n--){
		int id;
		int score;
		cin>>id>>score;
		Member::iterator big =member.upper_bound(score);
		Member::iterator small = big;
		small --;
		cout<<id<<" ";
		if(member.find(small->first)!=member.end()){
			int diff1 = big->first - score;
			int diff2= score - small->first;
			if(diff1<diff2)
				cout<<big->second<<endl;
			else
				cout<<small->second<<endl;

		}else
		cout<<big->second<<endl;
		member[score]=id;

	}
}