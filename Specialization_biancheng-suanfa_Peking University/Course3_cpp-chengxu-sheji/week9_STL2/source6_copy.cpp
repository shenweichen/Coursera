#include <iostream>
#include <cmath>
#include <set>
using namespace std;

//判断一个数是否为质数
bool isPrime(int iVal)
{
    for (int i = 2; i <= sqrt((float)iVal); i++)
    {
        if (iVal % i == 0)
            return false;
    }
    return true;
}

//得到一个数的质因数数目
int getFactor(int a)
{
    int sum = 0;
    int k = 0;
    for (int i = 2; i <= sqrt((float)a); i++)
    {
        if (a % i == 0)//即a/i = k，无余数
        {
            k = a / i;
            if (i != k && isPrime(k))
                sum += 1;
            if (isPrime(i))
            {
                sum += 1;
            }
        }
    }
    return sum;
}

//函数对象，判断大小
class Compare_Class_Up
{
public:
    bool operator()(const int& a, const int& b)
    {
        int sumA = 0, sumB = 0;
        sumA = getFactor(a);
        sumB = getFactor(b);
        if (sumA == sumB)
        {
            if (a < b)
            {
                return true;
            }
            else
                return false;
        }
        else if (sumA > sumB)
        {
            return false;
        }
        else
            return true;
    }
};


int main()
{
    set<int, Compare_Class_Up> queue;
    int n = 0;
    cin >> n;
    while (n--)
    {
        int num;
        for (int i = 0; i < 10; i++)
        {
            cin >> num;
            queue.insert(num);
        }

        int min = *(queue.begin());
        int max = *(queue.rbegin());
        queue.erase(max);
        queue.erase(min);
        cout << max << ' ' << min << endl;
    }
    return 0;
}