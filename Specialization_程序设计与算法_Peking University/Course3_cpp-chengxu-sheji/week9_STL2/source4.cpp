#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
using namespace std;
string  str[21];
int funN(const string &cmd);
string funS(const string &cmd);
string num("01234567890");
string Copy(){
	string temp ;
	int N,X,L;
	cin>>temp;
	N=funN(temp);
	cin>>temp;
	X=funN(temp);
	cin>>temp;
	L=funN(temp);
	return str[N].substr(X,L);
}
string Add(){
	string temp;
	string S1,S2;
	cin>>temp;
	S1=funS(temp);
	cin>>temp;
	S2=funS(temp);
	if(S1.find_first_not_of(num)==string::npos&&S2.find_first_not_of(num)==string::npos&&(atoi(S1.c_str())>=1&&atoi(S1.c_str())<=99999)&&(atoi(S2.c_str())>=1&&atoi(S2.c_str())<=99999)){
		//这里atoi("213saddsaf")的返回值仍然为213，要先判断你是否为全数字
		stringstream ss;
		ss<<atoi(S1.c_str())+atoi(S2.c_str());
		ss>>temp;
	}
	else
		temp =S1+S2;
	return temp;
}
int Find(){
	string temp;
	string S;
	int N;
	cin>>temp;
	S=funS(temp);
	cin>>temp;
	N = funN(temp);
	if(str[N].find(S)==string::npos)
		return S.length();
	else
		return str[N].find(S);

}
int rFind(){
	string temp;
	string S;
	int N;
	cin>>temp;
	S=funS(temp);
	cin>>temp;
	N = funN(temp);
	if(str[N].rfind(S)==string::npos)
		return S.length();
	else
		return str[N].rfind(S);
}
void Insert(){

	string temp,S;
	int N,X;
	cin>>temp;
	S=funS(temp);
	cin>>temp;
	N=funN(temp);
	cin>>temp;
	X=funN(temp);
	str[N].insert(X,S);
}
void Reset(){

	string temp,S;
	int N;
	cin>>temp;
	S= funS(temp);
	cin>>temp;
	N=funN(temp);
	str[N]=S; 
}
void Print(){
	string temp;
	int N;
	cin>>temp;
	N=funN(temp);
	cout<<str[N]<<endl;
}
void Printall(int n){
	for(int i=1;i<=n;i++)
		cout<<str[i]<<endl;
}
int funN(const string &cmd){
	if(cmd=="find")
		return Find();
	else if(cmd=="rfind")
		return rFind();
	else return atoi(cmd.c_str());
}

string funS(const string &cmd){
	if(cmd=="copy")
		return Copy();
	else if(cmd=="add")
		return Add();
	else
		return cmd;
}

int main(){
	int n;
	cin>>n;
	for(int i=1;i<=n;i++)
		cin>>str[i];
	while(cin>>str[0]){
		if(str[0]=="over")
			break;
		switch(str[0][0]){
			case 'i':{Insert();break;}
			case 'r':{Reset();break;}
			case 'p': {
				if (str[0] == "print")
					Print();
				else 
					Printall(n); 
				break; }
			
		}
	}
	
}