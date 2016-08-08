#include <iostream>
using namespace std;

int counts = 0;

//参数istack:栈中数据个数；参数ostack：待放入栈中的数据个数
//数据全部放入即表示生成一种序列
void fun(int istack, int ostack) {
  if (ostack == 0) {
    counts++;
  } else if (istack == 0) {
    fun(istack + 1, ostack - 1);
  } else {
    fun(istack - 1, ostack);
    fun(istack + 1, ostack - 1);
  }
}

int main() {
  int n;
  cin >> n;
  fun(0, n);
  cout << counts << endl;
  return 0;
}