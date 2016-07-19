/*#include <iostream>
using namespace std;*/

class CType
{
	int value;
public:
	CType()
	{

	}
	CType (int n) :value(n) {}
	void setvalue(int n){
		value = n;
	}
	CType& operator++(int i){
		static CType temp (value);//必须使用静态，否则返回时内存被释放了
		temp.setvalue(value);
		value = value*value;
		return  temp;
	}
	friend ostream & operator<<(ostream &out, CType & c) {
		out << c.value;
		return out;
	}
};

/*int main(int argc, char* argv[]) {
        CType obj;
        int n;
        cin>>n;
        while ( n ) {
                obj.setvalue(n);
                cout<<obj++<<" "<<obj<<endl;
                cin>>n;
        }
        return 0;
}*/