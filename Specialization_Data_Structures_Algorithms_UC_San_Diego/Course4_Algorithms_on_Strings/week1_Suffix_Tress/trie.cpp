#include <iostream>
#include <map>
#include <string>
#include <vector>

using std::map;
using std::vector;
using std::string;

typedef map<char, int> edges;
typedef vector<edges> trie;

trie build_trie(vector<string> &patterns) {
  trie t;
  t.resize(1);
  for (size_t i = 0; i < patterns.size(); i++) {
    int currentNode = 0;
    for (size_t j = 0; j < patterns[i].size(); j++) {
      char currentSymbol = patterns[i][j];
      if (t[currentNode].find(currentSymbol) != t[currentNode].end())
        currentNode = t[currentNode][currentSymbol];
      else {
        t.resize(t.size() + 1); //增加新结点
        t[currentNode][currentSymbol] = t.size() - 1;
        currentNode = t.size() - 1;
      }
    }
  }
  return t;
}

int main() {
  size_t n;
  std::cin >> n;
  vector<string> patterns;
  for (size_t i = 0; i < n; i++) {
    string s;
    std::cin >> s;
    patterns.push_back(s);
  }

  trie t = build_trie(patterns);
  for (size_t i = 0; i < t.size(); ++i) {
    for (const auto &j : t[i]) {
      std::cout << i << "->" << j.second << ":" << j.first << "\n";
    }
  }

  return 0;
}