#include <iostream>
#include <algorithm>
#include <cstring>
#include <stack>
using namespace std;
stack <char> st;
stack <int>  id;


int main() {
	char str[110];
	cin.getline(str,101,'\n');
	char boy = str[0];
	st.push(str[0]);
	id.push(0);
	for(int i=1;i<strlen(str);i++){
		if(str[i]!=st.top()){
			cout<<id.top()<<" "<<i<<endl;
			st.pop();
			id.pop();
		}else{
			st.push(str[0]);
			id.push(i);
		}
	}


}