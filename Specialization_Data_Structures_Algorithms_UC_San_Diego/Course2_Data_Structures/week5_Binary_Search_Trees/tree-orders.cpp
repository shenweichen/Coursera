#include <algorithm>
#include <iostream>
#include <vector>

using std::vector;
using std::ios_base;
using std::cin;
using std::cout;

class TreeOrders {
  int n;
  int root;
  vector<int> key;
  vector<int> left;
  vector<int> right;
  vector<bool> isRoot;

public:
  void read() {
    cin >> n;
    key.resize(n);
    left.resize(n);
    right.resize(n);
    isRoot.resize(n);
    for (int i = 0; i < n; i++) {
      cin >> key[i] >> left[i] >> right[i];
      if (left[i] != -1)
        isRoot[i] = 1;
      if (right[i] != -1)
        isRoot[i] = 1;
    }
    std::sort(isRoot.begin(), isRoot.end());
    root = isRoot[0];
  }
  void inorder(int root, vector<int> &r) {
    if (left[root] != -1)
      inorder(left[root], r);
    r.push_back(key[root]);
    if (right[root] != -1)
      inorder(right[root], r);
  }
  void preorder(int root, vector<int> &r) {
    r.push_back(key[root]);
    if (left[root] != -1)
      preorder(left[root], r);

    if (right[root] != -1)
      preorder(right[root], r);
  }
  void postorder(int root, vector<int> &r) {
    if (left[root] != -1)
      postorder(left[root], r);

    if (right[root] != -1)
      postorder(right[root], r);
    r.push_back(key[root]);
  }

  vector<int> in_order() {
    vector<int> result;
    // Finish the implementation
    // You may need to add a new recursive method to do that
    inorder(root, result);
    return result;
  }

  vector<int> pre_order() {
    vector<int> result;
    // Finish the implementation
    // You may need to add a new recursive method to do that
    preorder(root, result);
    return result;
  }

  vector<int> post_order() {
    vector<int> result;
    // Finish the implementation
    // You may need to add a new recursive method to do that
    postorder(root, result);
    return result;
  }
};

void print(vector<int> a) {
  for (size_t i = 0; i < a.size(); i++) {
    if (i > 0) {
      cout << ' ';
    }
    cout << a[i];
  }
  cout << '\n';
}

int main() {
  ios_base::sync_with_stdio(0);
  TreeOrders t;
  t.read();
  print(t.in_order());
  print(t.pre_order());
  print(t.post_order());
  return 0;
}
