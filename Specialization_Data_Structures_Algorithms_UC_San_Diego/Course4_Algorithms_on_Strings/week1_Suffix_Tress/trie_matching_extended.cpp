#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int const Letters = 4;
int const NA = -1;

struct Node {
  int next[Letters];
  bool patternEnd;

  Node() {
    fill(next, next + Letters, NA);
    patternEnd = false;
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

vector<int> solve(string text, int n, vector<string> patterns) {
  vector<int> result;

  // write your code here
  vector<Node> t(1);
  for (size_t i = 0; i < n; i++) { //构造Extend Trie树
    int currentNode = 0;
    for (size_t j = 0; j < patterns[i].size(); j++) {
      int symbol = letterToIndex(patterns[i][j]);
      if (t[currentNode].next[symbol] != NA) {
        currentNode = t[currentNode].next[symbol];
      } else {
        t.push_back(Node());
        t[currentNode].next[symbol] = t.size() - 1;
        currentNode = t.size() - 1;
      }
      if (j == patterns[i].size() - 1) //到达模式串末尾，添加标记
        t[currentNode].patternEnd = true;
    }
  }

  for (size_t i = 0; i < text.size(); i++) {
    size_t k = i;
    int symbol = letterToIndex(text[k]);
    int v = 0;
    while (true) {
      if (t[v].patternEnd) {
        result.push_back(i);
        break; //如果有多个串共享同一个前缀，只输出第一个即可
      } else if (t[v].next[symbol] != NA) {
        v = t[v].next[symbol];
        k++;
        if (k == text.size()) {
          if (t[v].patternEnd)
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
