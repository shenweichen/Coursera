#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

void check(const string &s, const vector<string> &dict) {
  vector<string> ans;
  for (size_t k = 0; k < dict.size(); k++) { //对dict每个元素检查
    vector<vector<int> > d(s.size() + 1, vector<int>(dict[k].size() + 1));
    //初始化edit矩阵
    for (size_t i = 0; i < s.size() + 1; i++)
      d[i][0] = i;
    for (size_t j = 0; j < dict[k].size() + 1; j++)
      d[0][j] = j;

    for (size_t i = 1; i <= s.size(); i++) {
      for (size_t j = 1; j <= dict[k].size(); j++) {
        int insetordel = min(d[i][j - 1] + 1, d[i - 1][j] + 1);
        int match = d[i - 1][j - 1];
        int mismatch = d[i - 1][j - 1] + 1;

        if (s[i - 1] == dict[k][j - 1]) //注意下标
          d[i][j] = min(insetordel, match);
        else
          d[i][j] = min(insetordel, mismatch);
      }
    }
    if (d[s.size()][dict[k].size()] == 0) { //找到完全匹配的直接返回
      cout << s << " is correct" << endl;
      return;
    }
    if (d[s.size()][dict[k].size()] == 1) { //编辑距离为1的加入ans
      ans.push_back(dict[k]);
    }
  }
  cout << s << ":";
  for (size_t i = 0; i < ans.size(); i++)
    cout << " " << ans[i];
  cout << endl;
}
int main() {
  vector<string> dict;
  string s;
  while (cin >> s) {
    if (s == "#")
      break;
    dict.push_back(s);
  }
  while (cin >> s) {
    if (s == "#")
      break;
    check(s, dict);
  }
  return 0;
}
