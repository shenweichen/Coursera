#include <iostream>
using namespace std;

int f(int m)//计算m中含有‘3’的个数
{
    int a[5];
    int count = 0;
    a[4] = m/10000;
    a[3] = m%10000/1000;
    a[2] = m%1000/100;
    a[1] = m%100/10;
    a[0] = m%10;
    for (int i = 0; i < 5; i++)
    {
        if (a[i] == 3)
        {
            count++;
        }
    }
    return count;
}
int main()
{
    int m, k;
    cin >> m >> k;
    if (m%19 == 0)//can be divided by 19
    {
        int count = f(m);
        if (count == k)
        {
            cout << "YES" << endl;
        }
        else
        {
            cout << "NO" << endl;
        }

    }
    else
    {
        cout << "NO" << endl;
    }
    return 0;
}