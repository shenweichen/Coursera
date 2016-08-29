#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

void shiftup(int x, vector<int> &heap) {
  while (x != 0) {
    size_t parent = (x - 1) >> 1;
    if (heap[x] < heap[parent])
      swap(heap[x], heap[parent]);
    x = parent;
  }
}
void shiftdown(int x, vector<int> &heap) {
  int minindex = x;
  int left = 2 * x + 1, right = 2 * x + 2;
  if (left < heap.size() && heap[left] < heap[minindex])
    minindex = left;
  if (right < heap.size() && heap[right] < heap[minindex])
    minindex = right;

  if (minindex != x) {
    swap(heap[x], heap[minindex]);
    shiftdown(minindex, heap);
  }
  return;
}
void insert(int key, vector<int> &heap) {
  heap.push_back(key);
  shiftup(heap.size() - 1, heap);
}
void remove(vector<int> &heap) {
  cout << heap[0] << endl;
  heap[0] = heap[heap.size() - 1];
  heap.resize(heap.size() - 1);
  shiftdown(0, heap);
}
int main() {
  int n;
  cin >> n;
  while (n--) {
    int t;
    cin >> t;
    vector<int> heap;
    while (t--) {
      int op;
      cin >> op;
      if (op == 1) {
        cin >> op;
        insert(op, heap);
      } else
        remove(heap);
    }
  }
  return 0;
}