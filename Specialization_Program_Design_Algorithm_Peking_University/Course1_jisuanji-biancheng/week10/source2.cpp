#include <iostream>
#include <cctype>
#include <cstring>
using namespace std;
int main(){
	char s1[80],s2[80];
	cin>>s1>>s2;

	for(int i=0;s1[i]!='\0';i++){
		if (isupper(s1[i])){
			s1[i]=tolower(s1[i]);
		}
	}

	for(int i=0;s2[i]!='\0';i++){
		if (isupper(s2[i])){
			s2[i]=tolower(s2[i]);
		}
	}
	int ans = strcmp(s1,s2);
	if(ans>0)
		cout<<">";
	else if (ans<0)
		cout<<"<";
	else
		cout<<"=";
	return 0;
}