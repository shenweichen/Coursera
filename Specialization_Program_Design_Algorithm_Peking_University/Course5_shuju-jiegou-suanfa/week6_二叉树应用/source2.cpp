#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
struct Node {
  int key;
  Node *left;
  Node *right;
  Node(int k = -1, Node *l = NULL, Node *r = NULL)
      : key(k), left(l), right(l){};
};
void add(Node *&root) { root->time++; }
void insert(Node *&root, int key) {
  if (root == NULL) {
    root = new Node(key);
    return;
  }
  if (key == root->key) {
    return; //这题要求重复元素只插入一个。。。坑了我很久
  }
  if (key < root->key)
    insert(root->left, key);
  else
    insert(root->right, key);
}
void preorder(Node *root) {
  if (root == NULL)
    return;
  for (size_t i = 0; i < root->time; i++)
    cout << root->key << " ";
  preorder(root->left);
  preorder(root->right);
}
int main() {
  int num;
  Node *root = NULL;
  string s;
  getline(cin, s);
  stringstream ss(s);
  while (ss >> num) {
    insert(root, num);
  }
  preorder(root);
  return 0;
}

//在不知道数据个数的情况下读入数据
/*
1、可以手动一个char一个char读，读到换行符时结束。缺点：需要考虑多个空格、不同的分隔符（\t,
\s）等等情况。
2、使用StringStream
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
int main() {
    // 1. Manually check
    while (true) {
        char c;
        while (c = getchar())
             if (c != ' ')
                break;
        if (c == '\n') break;
        int res = c-'0';
        while (c = getchar())
             if (c != ' ')
                res = res*10 + c-'0';
            else
                break;
        // do sth with res ....
    }
    // 2. Stream
    string s;
    getline(cin, s);
    stringstream ss(s);
    int res;
    while (ss >> res)
    {
        // do sth with res ....
    }
}

*/