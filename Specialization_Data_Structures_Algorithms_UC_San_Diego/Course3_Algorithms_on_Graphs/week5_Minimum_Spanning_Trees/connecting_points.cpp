#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using std::vector;
struct Node {
  int start, end;
  double dis;
  Node(int x, int y, double d) : start(x), end(y), dis(d){};
  bool operator<(const Node &a) const { return dis < a.dis; }
  // bool operator>(const Node &a) const { return dis < a.dis; }
  // 优先队列需要同时重载<和>
};
int Find(int x, vector<int> &parent) {
  if (x != parent[x])
    parent[x] = Find(parent[x], parent);
  return parent[x];
}
void Union(int x, int y, vector<int> &parent) {
  int left = Find(x, parent);
  int right = Find(y, parent);
  if (left != right) {
    parent[right] = left;
  }
}
double minimum_distance(vector<int> x, vector<int> y) {
  double result = 0.;
  // write your code here
  vector<int> parent(x.size());
  vector<Node> q;                       //按权值大小的优先队列
  for (size_t i = 0; i < x.size(); i++) //初始化并查集
    parent[i] = i;
  for (size_t i = 0; i < x.size(); i++) {
    for (size_t j = i + 1; j < x.size(); j++) {
      double dis =
          sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]));
      q.push_back(Node(i, j, dis));
    }
  }
  std::sort(q.begin(), q.end()); //将边按权值递增排序

  for (size_t i = 0; i < q.size(); i++) {
    Node v = q[i];
    if (Find(v.start, parent) !=
        Find(v.end, parent)) { //如果当前是一条连接两个集合最小边
      result += v.dis;         //将这条边加入到MST中
      Union(v.start, v.end, parent); //合并集合
    }
  }
  return result;
}

int main() {
  size_t n;
  std::cin >> n;
  vector<int> x(n), y(n);
  for (size_t i = 0; i < n; i++) {
    std::cin >> x[i] >> y[i];
  }
  std::cout << std::setprecision(10) << minimum_distance(x, y) << std::endl;
  return 0;
}
