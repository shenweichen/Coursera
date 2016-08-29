#include <cstdio>
#include <iostream>
#include <stack>
#include <vector>

using namespace std;

struct Node {
  int key;
  int left;
  int right;
  Node(int key) : key(key), left(-1), right(-1){};
};
vector<int> in;
vector<int> post;
vector<Node> Tree;
int createTree(int inL, int inR, int postL, int postR) {
  if (postL > postR)
    return -1;
  Node a(post[postR]);
  int root = Tree.size();
  Tree.push_back(a);
  int k;
  for (k = inL; k <= inR; k++) {
    if (in[k] == post[postR])
      break;
  }
  int numLeft = k - inL;
  Tree[root].left = createTree(inL, k - 1, postL, postL + numLeft - 1);
  Tree[root].right = createTree(k + 1, inR, postL + numLeft, postR - 1);

  return root;
}

void Preorder(int x) {

  stack<int> st;
  while (true) {
    while (x != -1) {
      cout << Tree[x].key << " ";
      if (Tree[x].right != -1)
        st.push(Tree[x].right);
      x = Tree[x].left;
    }
    if (st.empty())
      return;
    x = st.top();
    st.pop();
  }
  return;
}
int main() {

  int temp;
  while (cin >> temp) {
    in.push_back(temp);
    if (cin.get() != ' ')
      break;
  }
  while (cin >> temp) {
    post.push_back(temp);
    if (cin.get() != ' ')
      break;
  }
  int root = createTree(0, in.size() - 1, 0, post.size() - 1);
  Preorder(root);

  return 0;
}