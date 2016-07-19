/*#include <iostream>
#include <map>
using namespace std;*/

class A
{
	
public:
	static int count;
	A(){count++;}
	A(int n){count++;}
	virtual ~A(){
		cout<<"A::destructor"<<endl;}
	void operator delete(void* a)/*使用了delete释放某个重载了delete的类的对象空间时，先调用类的析构函数，
    然后再调用重载的delete函数。*/
    {
        //count--;
    }

};
class B:public A
{
public:
	B():A(){}
	B(int n):A(n){}
	B(B &b){};
	 ~B(){
	 	count--;
		cout<<"B::destructor"<<endl;}
	
};
/*int A::count = 0;
void func(B b) { }
int main()
{
        A a1(5),a2;
        cout << A::count << endl;
        B b1(4);
        cout << A::count << endl;
        func(b1);
        cout << A::count << endl;
        A * pa = new B(4);
        cout << A::count << endl;
        delete pa;
        cout << A::count << endl;
        return 0;
}
*/