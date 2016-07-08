#include <iostream>
#include <cstring>
using namespace std;

int main(){
	int n;
	cin>>n;
	while(n--){
		char str[256];
		cin>>str;
		int l = strlen(str);
		for(int i=0;i<l;i++){
			switch(str[i]){
				case 'A':cout<<'T';break;
				case 'T':cout<<'A';break;
				case 'G':cout<<'C';break;
				case 'C':cout<<'G';break;
			}
		}
		cout<<endl;
	}
}

