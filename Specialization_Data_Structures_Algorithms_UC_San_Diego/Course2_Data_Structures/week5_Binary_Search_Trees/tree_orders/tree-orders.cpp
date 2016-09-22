#include <algorithm>
#include <iostream>
#include <stack>
#include <vector>

using std::vector;
using std::ios_base;
using std::cin;
using std::cout;
using std::stack;

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
  void iter_inorder(int root, vector<int> &r) { // LNR 迭代中序遍历
    stack<int> st;
    int x = root;
    while (true) {
      while (x != -1) { //寻找最左下结点，沿途路径结点入栈
        st.push(x);
        x = left[x];
      }
      if (st.empty())
        break;
      x = st.top();        //
      r.push_back(key[x]); //访问
      st.pop();
      x = right[x]; //转向右子树，若右子树为空，则下次循环会弹出栈顶元素继续访问
    }
  }
  void recur_inorder(int root, vector<int> &r) {
    if (left[root] != -1)
      recur_inorder(left[root], r);
    r.push_back(key[root]);
    if (right[root] != -1)
      recur_inorder(right[root], r);
  }
  void iter_preorder(int root, vector<int> &r) { // NLR 迭代先序遍历
    stack<int> st;
    int x = root;
    while (true) {
      while (x != -1) {      //左孩子不空持续向左下访问
        r.push_back(key[x]); //访问
        if (right[x] != -1)  //若右孩子不空
          st.push(right[x]); //右子树根入栈
        x = left[x];         //访问下一个左孩子
      }
      if (st.empty())
        break;
      x = st.top();
      st.pop();
    }
  }
  void recur_preorder(int root, vector<int> &r) {
    r.push_back(key[root]);
    if (left[root] != -1)
      recur_preorder(left[root], r);

    if (right[root] != -1)
      recur_preorder(right[root], r);
  }
  void iter_postorder(int root, vector<int> &r) { // LRN  该算法还有问题。
    stack<int> st;
    int x = root;
    if (x != -1)
      st.push(x);
    while (!st.empty()) {
      if (left[st.top()] != x &&
          right[st.top()] != x) { //若栈顶非当前结点之父（则其必为右兄），
        //此时需在以其右兄为根的子树中寻找HLVFL
        while ((x = st.top()) != -1) {
          if (left[x] != -1) { //尽可能向左
            if (right[x] != -1)
              st.push(right[x]);
            st.push(left[x]);
          } else
            st.push(right[x]);
        }
        st.pop(); //返回之前，弹出栈顶空元素
      }
      x = st.top();
      r.push_back(key[x]); //访问元素
      st.pop();
    }
  }
  void recur_postorder(int root, vector<int> &r) {
    if (left[root] != -1)
      recur_postorder(left[root], r);

    if (right[root] != -1)
      recur_postorder(right[root], r);
    r.push_back(key[root]);
  }

  vector<int> in_order() {
    vector<int> result;
    // Finish the implementation
    // You may need to add a new recursive method to do that
    // recur_inorder(root, result);
    iter_inorder(root, result);
    return result;
  }

  vector<int> pre_order() {
    vector<int> result;
    // Finish the implementation
    // You may need to add a new recursive method to do that
    // recur_preorder(root, result);
    iter_preorder(root, result);
    return result;
  }

  vector<int> post_order() {
    vector<int> result;
    // Finish the implementation
    // You may need to add a new recursive method to do that
    // recur_postorder(root, result);
    iter_postorder(root, result);
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
