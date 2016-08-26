#include <iostream>
#include <cstring>
using namespace std;

int main() {
	char str[11];
	char substr[4];
	while(cin>>str>>substr){
		int l = strlen(str);
		int index = 0;
		int ascii = -1;
		for(int i=0;i<l;i++){
			if(str[i]>ascii){
				ascii = str[i];
				index = i;
			}
		}

		for(int i=0;i<l;i++){
			cout<<str[i];
			if(i==index){
				for(int j =0 ;j<strlen(substr);j++)
					cout<<substr[j];
			}
		}
		cout<<endl;
	}
}