/*#include <iostream>
using namespace std;*/
// 在此处补充你的代码
template <class T>
class CArray3D{
  
public:
    template <class T1>
    class CArray2D
    {
    	T1 *a1;
    	int row;
    	int column;
    public:
    	CArray2D():a1(NULL),row(0),column(0){}
    	CArray2D(int paraRow, int paraColumn):row(paraRow),column(paraColumn) { a1 = new T1[row*column];}
    	void set(int paraRow, int paraColumn)
        {
            row = paraRow;
            column = paraColumn;
            a1 = new T1[row*column];
        }

    	~CArray2D(){if(a1) delete [] a1;}
    	T1* operator[](int i){return a1+column*i;}

    };
private:
	CArray2D<T> *ptr;
public:
	CArray3D():ptr(NULL){}
	CArray3D(int z,int y,int x){
		ptr = new CArray2D<T>[z];
		for(int m=0;m<z;m++){
			ptr[m].set(y,x);
		}
	}
	CArray2D<T>& operator[](int i){return ptr[i];}
	~CArray3D(){if(ptr) delete [] ptr;}
};/*
int main()
{
    CArray3D<int> a(3,4,5);
    int No = 0;
    for( int i = 0; i < 3; ++ i )
        for( int j = 0; j < 4; ++j )
            for( int k = 0; k < 5; ++k )
                a[i][j][k] = No ++;
    for( int i = 0; i < 3; ++ i )
        for( int j = 0; j < 4; ++j )
            for( int k = 0; k < 5; ++k )
                cout << a[i][j][k] << ",";
    return 0;
}*/