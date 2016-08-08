#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

using namespace std;
int main() {
  int n, m;
  int count = 1, loc = 0;
  cin >> n >> m;
  vector<int> hash(n, 1);
  fill(hash.begin(), hash.begin() + n, 1);
  while (n != 1) {
    if (hash[loc] == 1) {
      if (count < m) {
        count++;
      } else {
        hash[loc] = 0;
        n--;
        count = 1;
      }
    }
    while (n == 1 && hash[loc] == 0)
      loc = (loc + 1) % hash.size();
    loc = (loc + 1) % hash.size();
  }
  cout << loc << endl;

  return 0;
}