#include <iostream>
#include <queue>
#include <stack>
#include <vector>
using namespace std;

struct Node {
  int weight;
  Node *left;
  Node *right;
  Node(int w, Node *l = NULL, Node *r = NULL) : weight(w), left(l), right(r){};
  Node() : left(NULL), right(NULL){}; //无参构造函数主要用于vector?
  bool operator<(Node b) { return weight < b.weight; }
  bool isLeaf() { return left == NULL && right == NULL; }
};
/*
class Mycmp {
public:
  bool operator()(const Node &a, const Node &b) { return a.weight > b.weight; }
  //优先队列返回最大元素，即<右边的元素，这里是小顶堆，应该让小的元素在右边？？
};
*/

//以下是自己实现的优先队列
template <class T> class my_priority_queue {
public:
  T top() {
    T ans = heap[0];
    return ans;
  }
  size_t size() { return heap.size(); }
  void pop() {
    heap[0] = heap[heap.size() - 1];
    heap.resize(heap.size() - 1);
    siftDown(0);
  }
  void push(T t) {
    heap.push_back(t);
    siftUp(heap.size() - 1);
  }

private:
  vector<T> heap;
  void siftUp(int x) {
    while (x != 0) {
      int parent = (x - 1) >> 1;
      if (heap[x] < heap[parent])
        swap(heap[x], heap[parent]);
      x = parent;
    }
  }
  void siftDown(int x) {
    int minIndex = x;
    int left = 2 * x + 1;
    int right = 2 * x + 2;
    if (left < heap.size() && heap[left] < heap[minIndex])
      minIndex = left;
    if (right < heap.size() && heap[right] < heap[minIndex])
      minIndex = right;
    if (minIndex != x) {
      swap(heap[minIndex], heap[x]);
      siftDown(minIndex);
    }
  }
};

int preorder(Node *root) {
  int weight = 0;
  int height = 0;
  stack<Node *> st;
  stack<int> h;
  while (true) {
    while (!root->isLeaf()) {
      height++;
      if (root->right != NULL) {
        st.push(root->right);
        h.push(height); //记录回溯时的高度
      }

      root = root->left;
    }
    weight += (height * root->weight);
    if (st.empty())
      break;
    root = st.top();
    height = h.top();
    st.pop();
    h.pop();
  }
  return weight;
}
int main() {
  int n;
  vector<int> weight;
  cin >> n;
  // priority_queue<Node, vector<Node>, Mycmp> heap;
  my_priority_queue<Node> heap; //使用自己的的优先队列
  for (size_t i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    heap.push(Node(temp)); //使用筛选法建堆比一个一个插入效率更高
  }
  vector<Node> Huffman(2 * n - 1); // vector会动态扩充容量，如果不事先指定大小
  //那么以前引用的指针地址会失效
  int H_size = 0;
  while (heap.size() > 1) {
    Huffman[H_size] = heap.top();
    heap.pop();
    Node *l = &Huffman[H_size++];

    Huffman[H_size] = heap.top();
    heap.pop();
    Node *r = &Huffman[H_size++];

    Node p(l->weight + r->weight, l, r);
    heap.push(p);
  }
  Huffman[H_size] = heap.top();
  Node *root = &Huffman[H_size];
  int ans = preorder(root);
  cout << ans << endl;
  return 0;
}