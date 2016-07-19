/*#include <iostream>
#include <set>
#include <iterator>
#include <algorithm>
using namespace std;*/

class A
{
	int age;
	char type;
public:
	A(int k,char c = 'A'):age(k),type(c){}
	int getage(){
		return age;
	}
	char gettype(){
		return type;
	}
	
};
class B:public A
{
public:
	B(int k):A(k,'B'){}
	
};
class Comp
{
public:
	bool operator()(A* a,A* b){
		if(a->getage()<b->getage())
			return true;
		else
			return false;
	}
	
};
void Print(A* a){
	cout<<a->gettype()<<" "<<a->getage()<<endl;
}
/*int main()
{

        int t;
        cin >> t;
        set<A*,Comp> ct;
        while( t -- ) {
                int n;
                cin >> n;
                ct.clear();
                for( int i = 0;i < n; ++i)	{
                        char c; int k;
                        cin >> c >> k;

                        if( c == 'A')
                                ct.insert(new A(k));
                        else
                                ct.insert(new B(k));
                }
                for_each(ct.begin(),ct.end(),Print);
                cout << "****" << endl;
        }
}*/