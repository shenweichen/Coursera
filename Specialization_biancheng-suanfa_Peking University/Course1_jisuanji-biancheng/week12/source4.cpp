#include <iostream>
using namespace std;

int main() {
	int a,b;
	char op;
	cin>>a>>b>>op;
	switch(op){
		case '+':cout<<a+b;break;
		case '-':cout<<a-b;break;
		case '*':cout<<a*b;break;
		case '/':
			if(b==0)
				cout<<"Divided by zero!";
			else
				cout<<a/b;
			break;
		default:
			cout<<"Invalid operator!";
	}
}