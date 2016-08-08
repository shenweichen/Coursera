#include <iostream>
#include <iterator>
#include <map>
using namespace std;
class Mygreater {
public:
  bool operator()(const int &a, const int &b) { return a > b; }
};
typedef map<int, int, Mygreater> MAP;
int main() {
  int n;
  cin >> n;
  while (n--) {
    MAP a;
    int factor, pow;
    while (cin >> factor >> pow && pow >= 0) {
      a[pow] = a.find(pow) != a.end() ? a[pow] + factor : factor;
    }
    while (cin >> factor >> pow && pow >= 0) {
      a[pow] = a.find(pow) != a.end() ? a[pow] + factor : factor;
    }
    for (MAP::iterator it = a.begin(); it != a.end(); it++) {
      if (it->second != 0)
        cout << "[ " << it->second << " " << it->first << " ] ";
    }
    cout << endl;
  }
  return 0;
}
