/*#include<iostream>
using namespace std;
class Number {
public:
    int num;
    Number(int n=0): num(n) {}
*/
    Number operator*(const Number& b){
    	Number temp;
    	temp.num = this->num*b.num;
    	return temp;
    }
    operator int(){
    	return this->num;
    }
/*
};
	
int main() {
    Number n1(10), n2(20);
    Number n3;n3 = n1*n2;
    cout << int(n3) << endl;
    return 0;
}*/