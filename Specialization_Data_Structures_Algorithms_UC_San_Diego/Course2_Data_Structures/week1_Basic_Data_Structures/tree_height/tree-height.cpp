#include <algorithm>
#include <iostream>
#include <vector>


class TreeHeight {
  int n;
  // std::vector<int> parent;
  std::vector<std::vector<int>> children;
  int root;

public:
  void read() {
    std::cin >> n;
    children.resize(n);
    for (int i = 0; i < n; i++) {
      int parent;
      std::cin >> parent;
      if (parent == -1)
        root = i;
      else
        children[parent].push_back(i);
    }
  }
  int compute(int root) {
    int max = 0;
    for (int i = 0; i < children[root].size(); i++) {
      max = std::max(max, compute(children[root][i]));
    }
    return max + 1;
  }
  int compute_height() {
    // Replace this code with a faster implementation

    return compute(root);
  }
};

int main() {
  std::ios_base::sync_with_stdio(0);
  TreeHeight tree;
  tree.read();
  std::cout << tree.compute_height() << std::endl;
  return 0;
}
