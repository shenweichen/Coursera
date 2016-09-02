#include <iostream>
#include <vector>

using std::vector;
using std::pair;
bool isDAG = true;

void dfs(int start, vector<vector<int> > &adj, vector<int> &visited) {
  visited[start] = -1;
  for (size_t i = 0; i < adj[start].size() && isDAG; i++) {
    if (visited[adj[start][i]] == 0) {
      dfs(adj[start][i], adj, visited);
    } else if (visited[adj[start][i]] == -1) {
      isDAG = false;
      return;
    }
  }
  visited[start] = 1;
}
int acyclic(vector<vector<int> > &adj) {
  vector<int> visited(adj.size(), 0);
  for (size_t i = 0; i < adj.size(); i++) {
    if (visited[i] == 0) {
      dfs(i, adj, visited);
    }
    if (!isDAG)
      return 1;
  }
  return 0;
}

int main() {
  size_t n, m;
  std::cin >> n >> m;
  vector<vector<int> > adj(n, vector<int>());
  for (size_t i = 0; i < m; i++) {
    int x, y;
    std::cin >> x >> y;
    adj[x - 1].push_back(y - 1);
  }

  std::cout << acyclic(adj);
  return 0;
}
