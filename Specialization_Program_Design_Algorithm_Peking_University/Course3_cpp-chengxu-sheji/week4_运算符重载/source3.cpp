/*#include <iostream>
#include <cstring>
using namespace std;*/
class Array2
{
	int *ptr;
	int row;
	int column;
public:
	Array2(int i,int j){
		row = i;
		column = j;
		ptr = new int[row*column];
	};
	Array2(){
		ptr=NULL;
	};
	~Array2() {
		if (ptr )
			delete[] ptr;
	};

	Array2& operator=(const Array2 &A){
		if(ptr)
			delete []ptr;
		row = A.row;
		column=A.column;
		ptr = new int[row*column];
		memcpy(ptr,A.ptr,sizeof(int)*row*column);

		return *this;
	}
	int* operator[](int i){//重载的实际上是第二维的[]， 第一维的[]直接调用int型一维数组的定义

	
			return ptr+column*i;
	};
	int operator()(int i,int j){
		return ptr[i*column+j];
	}
	
	
};
/*
int main() {

    Array2 a(3,4);
    int i,j;
    for(  i = 0;i < 3; ++i )
        for(  j = 0; j < 4; j ++ )
            a[i][j] = i * 4 + j;
    for(  i = 0;i < 3; ++i ) {
        for(  j = 0; j < 4; j ++ ) {
            cout << a(i,j) << ",";
        }
        cout << endl;
    }
    cout << "next" << endl;
    Array2 b;     b = a;
    for(  i = 0;i < 3; ++i ) {
        for(  j = 0; j < 4; j ++ ) {
            cout << b[i][j] << ",";
        }
        cout << endl;
    }
    return 0;
}*/