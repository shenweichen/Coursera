#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

string InverseBWT(const string &bwt) {
  string text = "";

  // write your code here
  vector<string> BWTM(bwt.size());
  string temp;
  temp.resize(bwt.size());
  std::partial_sort_copy(bwt.begin(), bwt.end(), temp.begin(), temp.end());
  for (size_t j = 0; j < BWTM.size(); j++) {
    BWTM[j].push_back(temp[j]);
  }
  for (size_t i = 1; i < bwt.size(); i++) {
    for (size_t j = 0; j < BWTM.size(); j++) {
      BWTM[j].insert(BWTM[j].begin(), bwt[j]); //每次将bwt的元素加在BWTM的前面
    }
    sort(BWTM.begin(), BWTM.end());
  }
  text.assign(BWTM[0], 1, BWTM[0].size() - 1);
  text += "$";
  return text;
}

int main() {
  string bwt;
  cin >> bwt;
  cout << InverseBWT(bwt) << endl;
  return 0;
}
