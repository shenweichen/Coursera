#include <iostream>
using namespace std;
int main(){
	char s[500];
	cin.getline(s,500);
	int length = 0;
	int max = -1;
	int i=0;
	int index=1;
	while(s[i]!='.'){
		
		if(s[i]!=' '){
			length++;
			i++;
		}
		else{
			if(length>max){
				max = length;
				index = i-1;
			}
			length =0;
			i++;
		}
		if(s[i]=='.'&&length>max){
			max = length;
				index = i-1;
		}
	}
	for(int i=index-max+1;i<=index;i++)
		cout<<s[i];

}