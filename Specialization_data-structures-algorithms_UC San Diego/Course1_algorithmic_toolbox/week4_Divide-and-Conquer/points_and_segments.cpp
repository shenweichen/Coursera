//区间点覆盖问题
#include <algorithm>
#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using std::vector;
using std::unordered_map;
/*
//以下为线段树
class Node {
public:
  friend Node *build(int l, int r);
  Node(int l, int r) { //初始化
    right = r;
    left = l;
    covertimes = 0;
    leftchild = NULL;
    rightchild = NULL;
  };
  void Insert(int l, int r) {
    if (l <= left &&
        r >= right) { //使用查询点的区间构造线段树，实际区间可能超过构造的区间
      covertimes++;
      return;
    }
    if (left == right)
      return;
    int mid = (left + right) >> 1;
    if (mid >= r)
      leftchild->Insert(l, r);
    else if (mid < l)
      rightchild->Insert(l, r);
    else {
      leftchild->Insert(l, mid);
      rightchild->Insert(mid + 1, r);
    }
  }
  int Search(int point) {
    int sum = 0;
    if (point < left || point > right) {
      return sum;
    }
    sum += covertimes;
    if (left == right) {
      return sum;
    }
    int mid = (left + right) >> 1;
    if (point <= mid)
      sum += (leftchild->Search(point));
    else
      sum += (rightchild->Search(point));
    return sum;
  }

private:
  int left;
  int right;
  int covertimes;
  Node *leftchild;
  Node *rightchild;
};

Node *build(int l, int r) {
  Node *root = new Node(l, r);
  if (l + 1 <= r) {
    int mid = (r + l) >> 1;
    root->leftchild = build(l, mid);
    root->rightchild = build(mid + 1, r);
  }
  return root;
}
//以上为线段树

inline int bsfirst(const vector<int> &p, int l, int val) {
  int r = p.size() - 1;
  while (l < r) {
    int mid = (l + r) >> 1;
    if (p[mid] >= val) { // val在左边区间
      r = mid;
    } else
      l = mid + 1;
  }
  if (p[l] >= val) //找到第一个大于等于val
    return l;
  else
    return -1;
}
inline int bslast(const vector<int> &p, int r, int val) {
  int l = 0;
  while (l < r - 1) {
    int mid = (l + r) >> 1;
    if (p[mid] <= val) { // val在左边区间
      l = mid;
    } else
      r = mid - 1;
  }
  if (p[l] <= val) //找到第一个大于等于val
    return l;
  else
    return -1;
}
vector<int> fast_count_segments2(vector<int> starts, vector<int> ends,
                                 vector<int> points) {
  vector<int> cnt(points.size());
  // write your code here
  unordered_map<int, int> hash;
  for (int i = 0; i < points.size(); i++) {
    hash[points[i]] = i;
  }
  // vector<int> org(points);
  std::sort(points.begin(), points.end());
  int u, v;
  int last_u = 0;
  int last_v = points.size() - 1;

  for (size_t i = 0; i < starts.size(); i++) {
    u = bsfirst(points, last_u, starts[i]);
    v = bslast(points, last_v, ends[i]);
    if (u != -1 && v != -1) {
      last_u = u;
      last_v = v;
      while (u <= v) {
        cnt[hash[points[u]]]++;
        u++;
      }
    }
  }

  return cnt;
}

vector<int> fast_count_segments(vector<int> starts, vector<int> ends,
                                vector<int> points, int l, int r) {
  vector<int> cnt(points.size());
  // write your code here

  Node *root = build(l, r); //使用查询点的区间来初始化，减少空间消耗

  for (size_t i = 0; i < starts.size(); i++) {

    root->Insert(std::max(l, starts[i]),
                 std::min(r, ends[i])); //避免有超过区间的值
  }
  for (size_t i = 0; i < points.size(); i++) {
    cnt[i] = root->Search(points[i]);
  }

  return cnt;
}
*/
vector<int> fast_count_segments3(vector<int> starts, vector<int> ends,
                                 vector<int> points) {
  vector<int> cnt(points.size());
  // write your code here
  std::sort(starts.begin(), starts.end());
  std::sort(ends.begin(), ends.end());
  unordered_map<int, int> hash;
  for (int i = 0; i < points.size(); i++) {
    hash[points[i]] = i;
  }
  std::sort(points.begin(), points.end());

  int k = 0;
  while (points[k] < starts[0] && k < points.size()) {
    cnt[hash[points[k]]] = 0;
    k++;
  }
  int ks = k;

  vector<int> dpl(points.size());
  vector<int> dpr(points.size());
  int sum = 0;
  for (int i = 0; i < starts.size() && k < points.size();) {
    if (starts[i] <= points[k]) {
      dpl[k]++;
      i++;
    } else {
      k++;
      dpl[k] = dpl[k - 1];
    }
  }
  while (k + 1 <= points.size()) {
    dpl[k + 1] = dpl[k];
    k++;
  }
  k = ks;
  for (int i = 0; i < starts.size() && k < points.size();) {
    if (ends[i] < points[k]) {
      dpr[k]++;
      i++;
    } else {
      k++;
      dpr[k] = dpr[k - 1];
    }
  }
  while (k + 1 < points.size()) {
    dpr[k + 1] = dpr[k];
    k++;
  }
  for (int t = ks; t < points.size(); t++) {
    cnt[hash[points[t]]] = dpl[t] - dpr[t];
  }

  return cnt;
}
vector<int> naive_count_segments(vector<int> starts, vector<int> ends,
                                 vector<int> points) {
  vector<int> cnt(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    for (size_t j = 0; j < starts.size(); j++) {
      cnt[i] += starts[j] <= points[i] && points[i] <= ends[j];
    }
  }
  return cnt;
}

int main() {
  int n, m;
  std::cin >> n >> m;
  vector<int> starts(n), ends(n);
  for (size_t i = 0; i < starts.size(); i++) {
    std::cin >> starts[i] >> ends[i];
  }
  vector<int> points(m);
  for (size_t i = 0; i < points.size(); i++) {
    std::cin >> points[i];
  }
  // use fast_count_segments
  vector<int> cnt = fast_count_segments3(starts, ends, points);
  vector<int> right = naive_count_segments(starts, ends, points);
  for (size_t i = 0; i < cnt.size(); i++) {
    std::cout << cnt[i] << ' ';
  }
  std::cout << std::endl;
  for (size_t i = 0; i < right.size(); i++) {
    std::cout << right[i] << ' ';
  }
  return 0;
}
