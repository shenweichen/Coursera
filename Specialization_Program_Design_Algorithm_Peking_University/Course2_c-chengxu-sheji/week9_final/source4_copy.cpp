#include <iostream>
#include <vector>
using namespace std;

int main()
{
    int n;
    cin >> n;
    vector<vector<int> > a(n, vector<int>(2));
    vector<int> resultx;
    vector<int> resulty;
    //输入数据
    for (int i = 0; i < n; i++)
    {
        cin >> a[i][0] >> a[i][1];
    }
    //根据定义找出符合要求的点
    int first = 0;
    for (int i = 0; i < n; i++)
    {
        int count = 0;
        for (int j = 0; j < n; j++)
        {
            if (j == i)     continue;
            if (a[j][0]>=a[i][0] && a[j][1]>=a[i][1])
            {
                break;
            }
            else
            {
                count++;
            }
            if (count == n-1)//yes
            {
                resultx.push_back(a[i][0]);
                resulty.push_back(a[i][1]);
            }
        }
    }
    //将符合要求的点排序
    for (int i = 0; i < resultx.size()-1; i++)
    {
        for (int j = i+1; j < resultx.size(); j++)
        {
            if (resultx[i]>resultx[j])
            {
                int tmpx, tmpy;
                tmpx = resultx[i];
                resultx[i] = resultx[j];
                resultx[j] = tmpx;
                tmpy = resulty[i];
                resulty[i] = resulty[j];
                resulty[j] = tmpy;
            }
        }
    }
    //输出结果
    cout << '(' << resultx[0] << ',' << resulty[0] << ')';
    for (int i = 1; i < resultx.size(); i++)
    {
        cout << ',' << '(' << resultx[i] << ',' << resulty[i] << ')';
    }
    cout << endl;
    return 0;
}