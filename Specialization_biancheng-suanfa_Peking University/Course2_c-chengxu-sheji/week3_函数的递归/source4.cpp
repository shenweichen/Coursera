#include <iostream>
#include <algorithm>
#include <cstring>
#include <stack>
using namespace std;


int main() {
			char str[110];
	while(cin.getline(str,101)){
	stack <char> st;
	cout<<str<<endl;
	stack <int>  id;
	int right[101] = { 0 };
	int left[101] = { 0 };
	for(int i=0;i<strlen(str);i++){
		if(str[i]=='('){
			st.push('(');
			id.push(i);
		}else if(str[i]==')'){
			if(!st.empty()){
				st.pop();
				id.pop();
			}
			else
				right[i]=1;
		}
	}

	while(!id.empty()){
		left[id.top()]=1;
		id.pop();
	}

	for(int i=0;i<strlen(str);i++){
		if(left[i]==1)
			cout<<"$";
		else if(right[i]==1)
			cout<<"?";
		else
			cout<<" ";
	}
	cout<<endl;

}
}