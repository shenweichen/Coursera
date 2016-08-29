#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

int const Letters = 4;
int const NA = -1;

struct Node {
  int next[Letters];

  Node() { fill(next, next + Letters, NA); }

  bool isLeaf() const {
    return (next[0] == NA && next[1] == NA && next[2] == NA && next[3] == NA);
  }
};

int letterToIndex(char letter) {
  switch (letter) {
  case 'A':
    return 0;
    break;
  case 'C':
    return 1;
    break;
  case 'G':
    return 2;
    break;
  case 'T':
    return 3;
    break;
  default:
    assert(false);
    return -1;
  }
}

vector<int> solve(const string &text, int n, const vector<string> &patterns) {
  vector<int> result;

  // write your code here
  vector<Node> t(1);
  for (size_t i = 0; i < n; i++) { //构造Trie树
    int currentNode = 0;
    for (size_t j = 0; j < patterns[i].size(); j++) {
      int currentSymbol = letterToIndex(patterns[i][j]);
      if (t[currentNode].next[currentSymbol] != NA)
        currentNode = t[currentNode].next[currentSymbol];
      else {
        t.push_back(Node());
        t[currentNode].next[currentSymbol] = t.size() - 1;
        currentNode = t.size() - 1;
      }
    }
  }
  for (size_t i = 0; i < text.size(); i++) { //进行匹配
    int k = i;
    int symbol = letterToIndex(text[k]);
    int v = 0;
    while (true) {
      if (t[v].isLeaf()) {
        result.push_back(i);
        break;
      } else if (t[v].next[symbol] != NA) {
        v = t[v].next[symbol];
        k++;
        if (k == text.size()) { //进行特判
          if (t[v].isLeaf())
            result.push_back(i);
          break;
        }
        symbol = letterToIndex(text[k]);

      } else
        break;
    }
  }

  return result;
}

int main(void) {
  string t;
  cin >> t;

  int n;
  cin >> n;

  vector<string> patterns(n);
  for (int i = 0; i < n; i++) {
    cin >> patterns[i];
  }

  vector<int> ans;
  ans = solve(t, n, patterns);

  for (int i = 0; i < (int)ans.size(); i++) {
    cout << ans[i];
    if (i + 1 < (int)ans.size()) {
      cout << " ";
    } else {
      cout << endl;
    }
  }
  return 0;
}
