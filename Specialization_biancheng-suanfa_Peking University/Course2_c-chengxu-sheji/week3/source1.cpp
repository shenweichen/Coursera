#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
int main() {
	char str[510];
	cin.getline(str,501,'\n');
	int len =strlen(str);
	int start = 0;
	for(int i=0;i<len;i++){
		if(str[i]==' '){
			reverse(str+start,str+i);
			while(str[i]==' ')
				i++;
			start = i;
		}
	}
	reverse(str+start,str+len);
	cout<<str;

}