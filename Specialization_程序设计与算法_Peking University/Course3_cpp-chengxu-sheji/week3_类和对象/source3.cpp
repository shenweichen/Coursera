/*#include <iostream>
using namespace std;
class Base {
public:
    int k;
    Base(int n):k(n) { }
	Base() {};
};

class Big  {
public:
    int v; 
    Base b;*/
	Big(int n) :v(n),b(n) {}
    Big(Big &B):v(B.v),b(B.b){}

/*};
int main()  {
    Big a1(5);    Big a2 = a1;
    cout << a1.v << "," << a1.b.k << endl;
    cout << a2.v << "," << a2.b.k << endl;
    return 0;
}
*/