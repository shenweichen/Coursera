#include <algorithm>
#include <ctime> //获取当前时间用
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using std::vector;
using std::string;

string itos(int i) // convert int to string
{
  std::stringstream s;
  s << i;
  return s.str();
}
bool stress_test(const string &ans, vector<string> a) {
  std::sort(a.begin(), a.end());
  while (std::next_permutation(a.begin(), a.end())) {
    std::stringstream rettemp;
    for (size_t i = 0; i < a.size(); i++) {
      rettemp << a[i];
    }

    string temp;
    rettemp >> temp;
    if (temp > ans) {
      std::cout << "error:" << ans << "diff:" << temp << std::endl;
      return false;
    }
  }
  return true;
}
bool cmp(const string &a, const string &b) {
  return a+b > b+a;
}
bool cmp2(const string &a, const string &b) {
  if (a.size() == b.size())
    return a > b;

  int pos = 0;
  bool islast = false;
  while (a[pos] == b[pos]) {
    pos++;
    if (pos == a.size() || pos == b.size()) {
      islast = true;
      break;
    }
  }             //找到第一个不相等的位置
  if (islast) { //有一个数组遍历结束时，元素全部相等，考察较长数组剩下的元素
    const string &longstr =
        a.size() > b.size() ? a : b; // longstr指向较长的数组
    string last =
        longstr.substr(pos, longstr.size() - pos); //取出长数组剩余元素
    int minsize = last.size() <= pos ? last.size() : pos;
    if (longstr == a) {
      return last.substr(0, minsize) > longstr.substr(0, minsize);
    } else {
      return last.substr(0, minsize) < longstr.substr(0, minsize);
    }
  } else { //两个数组都未遍历结束就找到了不相等的位置
    return a.substr(pos, a.size() - pos) > b.substr(pos, b.size() - pos);
  }
}
string largest_number(vector<string> a) {
  // write your code here
  std::sort(a.begin(), a.end(), cmp);
  std::stringstream ret;
  for (size_t i = 0; i < a.size(); i++) {
    ret << a[i];
  }
  string result;
  ret >> result;
  /* if (stress_test(result, a) == false) {
     system("pause");
   }*/
  return result;
}

int main() {

  int n;
  //  while (true) {
  srand((unsigned)time(0));
  // n = rand() % 6 + 1;
  std::cin >> n;
  vector<string> a(n);
  for (size_t i = 0; i < a.size(); i++) {
    std::cin >> a[i];
    // a[i] = itos(rand() % int(1000) + 1);
  }
  std::cout << largest_number(a) << std::endl;
  //}

  return 0;
}
