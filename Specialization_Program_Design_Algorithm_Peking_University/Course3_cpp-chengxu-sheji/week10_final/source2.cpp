/*#include <iostream>
using namespace std;
class Number {
public:
    int num;
    Number(int n): num(n) {
    }*/

    Number& value(){
    	return *this;
    }
   friend ostream& operator<<(ostream & out,const Number & a){
   		out<<a.num;
   		return out;
   }
   void operator=(int _num){
   	num=_num;
   }
   void operator+(const Number &a){
   	num+=a.num;
   }/*
};
int main() {
    Number a(2);
    Number b = a;
    cout << a.value() << endl;
    cout << b.value() << endl;
    a.value() = 8;
    cout << a.value() << endl;
    a+b;
    cout << a.value() << endl;
    return 0;
}*/