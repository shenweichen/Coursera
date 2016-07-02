#include <iostream>
using namespace std;
int a[100];
int main() {
    int n;
    cin >> n;
    for (int i = 0; i<n; i++)
        cin >> a[i];
    while (n--) { //常用的倒序计数循环，等价于while(n-->0)
        cout << a[n];
        if (n > 0) cout << " "; //如果不是最后一个数那么就要用空格分隔开
    }
    return 0;
}
//顺序输入倒序输出，但是如果我们要求通过修改数组本身实现逆序重放应该怎么做呢？