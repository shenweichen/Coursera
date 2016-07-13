#include <iostream>
#include <cstring>
using namespace std;
class Array2
{
	int *ptr=NULL;
	int row;
	int column;
	int *now=NULL;
	bool index;
public:
	Array2(int i,int j){
		row = i;
		column = j;
		ptr = new int[row*column];
		now = ptr;
		index = 0;
	};
	Array2(){

	};
	~Array2() {
		if (ptr != NULL)
			delete []ptr;
	};
	//Array2(const &A){
	//};
	Array2& operator= (int i){
		*now = i;
		return *this;
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
	Array2& operator[](int i){
		if(index ==0){
			index=1;
			now = ptr+column*i;
			return *this;
		}else{
			index = 0;
			now = now + i;
			return *this;
		}
	};
	int operator()(int i,int j){
		return *(ptr+column*i+j);
	}
	friend ostream & operator<<(ostream &o, const Array2 &A);
	
};
ostream & operator<<(ostream &o, const Array2 &A) {
	o << *(A.now);
	return o;
};
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
}