#include <iostream>
using namespace std;
int main(){

	char s[80];
	cin.getline(s,80);
	int num[5]={0};
	for(int i=0;s[i]!='\0';i++){
		switch(s[i]){
			case 'a':num[0]++;break;
			case 'e':num[1]++;break;
			case 'i':num[2]++;break;
			case 'o':num[3]++;break;
			case 'u':num[4]++;break;
		}
	}
	for(int i=0;i<5;i++){
		cout<<num[i]<<" ";
	}
	return 0;
}