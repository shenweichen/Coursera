#include <iostream>
#include <vector>

using std::vector;
using std::pair;
vector<bool> visited;

void DFS(int start, vector<vector<int>> &adj) {
  visited[start] = true;
  for (size_t i = 0; i < adj[start].size(); i++) {
    if (!visited[adj[start][i]])
      DFS(adj[start][i], adj);
  }
}
int number_of_components(vector<vector<int>> &adj) {
  int res = 0;
  // write your code here
  for (size_t i = 0; i < adj.size(); i++) {
    if (!visited[i]) {
      DFS(i, adj);
      res++;
    }
  }
  return res;
}

int main() {
  size_t n, m;
  std::cin >> n >> m;
  vector<vector<int>> adj(n, vector<int>());
  for (size_t i = 0; i < m; i++) {
    int x, y;
    std::cin >> x >> y;
    adj[x - 1].push_back(y - 1);
    adj[y - 1].push_back(x - 1);
  }
  visited.resize(n);
  std::cout << number_of_components(adj);
  return 0;
}
