#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;
  vector<int> ans(a.size() + b.size() + 1);
  int c = 0;
  for (int i = b.size() - 1; i >= 0; i--) {
    c = 0;
    int idx;
    for (int j = a.size() - 1; j >= 0; j--) {
      int temp = (b[i] - '0') * (a[j] - '0') + c;
      idx = b.size() - 1 - i + a.size() - 1 - j;
      ans[idx] += temp;
      c = ans[idx] / 10; //处理进位
      ans[idx] %= 10;
    }
    while (c) { //处理最后一次的进位
      ans[idx + 1] = c % 10;
      idx++;
      c /= 10;
    }
  }

  int len = ans.size();
  while (ans[len - 1] == 0) //去除前导0
    len--;
  ans.resize(len);
  reverse(ans.begin(), ans.end());
  for (int i = 0; i < ans.size(); i++)
    cout << ans[i];
  cout << endl;
  return 0;
}