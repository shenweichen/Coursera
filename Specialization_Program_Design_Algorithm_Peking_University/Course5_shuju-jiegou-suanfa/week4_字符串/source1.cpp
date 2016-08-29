#include <cstring>
#include <iostream>
#include <map>
#include <stack>
#include <string>

using namespace std;
map<string, int> mp;

void init() {
  mp["negative"] = -1;
  mp["zero"] = 0;
  mp["one"] = 1;
  mp["two"] = 2;
  mp["three"] = 3;
  mp["four"] = 4;
  mp["five"] = 5;
  mp["six"] = 6;
  mp["seven"] = 7;
  mp["eight"] = 8;
  mp["nine"] = 9;
  mp["ten"] = 10;
  mp["eleven"] = 11;
  mp["twelve"] = 12;
  mp["thirteen"] = 13;
  mp["fourteen"] = 14;
  mp["fifteen"] = 15;
  mp["sixteen"] = 16;
  mp["seventeen"] = 17;
  mp["eighteen"] = 18;
  mp["nineteen"] = 19;
  mp["twenty"] = 20;
  mp["thirty"] = 30;
  mp["forty"] = 40;
  mp["fifty"] = 50;
  mp["sixty"] = 60;
  mp["seventy"] = 70;
  mp["eighty"] = 80;
  mp["ninety"] = 90;
  mp["hundred"] = 100;
  mp["thousand"] = 1000;
  mp["million"] = 1000000;
}
int main() {
  init();
  while (true) {
    char s[10000];
    stack<string> st;
    cin.getline(s, 10000);
    if (strlen(s) == 0 || s[0] == ' ') //退出条件
      break;
    int start = 0;
    for (int i = 0; s[i] != '\0'; i++) {
      if (s[i] == ' ') {
        st.push(string(s + start, s + i));
        start = i + 1;
      }
    }
    st.push(string(s + start, s + strlen(s)));

    int sum = 0;
    int num = 0;
    int factor = 1;
    int p = 1;
    bool isnegative = false;
    while (!st.empty()) {
      string temp = st.top();
      st.pop();
      if (mp[temp] == 100 || mp[temp] == 1000 || mp[temp] == 1000000) { //是单位
        if (mp[temp] > factor) { //大单位
          sum += (num * factor);
          num = 0;
          p = 1;
          factor = mp[temp];
        } else //小单位
          p = mp[temp];
      } else if (mp[temp] == -1) { //是符号
        isnegative = true;
      } else { //是数字
        num += (mp[temp] * p);
      }
    }
    sum += (num * factor); //处理最后情况
    if (isnegative)
      cout << -1 * sum << endl;
    else
      cout << sum << endl;
  }
  return 0;
}