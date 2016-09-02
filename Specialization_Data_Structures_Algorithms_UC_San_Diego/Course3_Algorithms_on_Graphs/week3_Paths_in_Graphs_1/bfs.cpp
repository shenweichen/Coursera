#include <iostream>
#include <vector>
#include <queue>

using std::vector;
using std::queue;

int distance(vector<vector<int> > &adj, int s, int t) {
  //write your code here
  const int INF = -1;
  vector <int> dis(adj.size(),INF);
  dis[s]=0;
  queue<int> q;
  q.push(s);
  while(!q.empty()){
    int cur = q.front();
    q.pop();
    for(size_t i =0;i<adj[cur].size();i++){
      int next = adj[cur][i];
      if(dis[next]==INF){
        q.push(next);
        dis[next]=dis[cur]+1;
      }
    }
  }
  return dis[t];
}

int main() {
  int n, m;
  std::cin >> n >> m;
  vector<vector<int> > adj(n, vector<int>());
  for (int i = 0; i < m; i++) {
    int x, y;
    std::cin >> x >> y;
    adj[x - 1].push_back(y - 1);
    adj[y - 1].push_back(x - 1);
  }
  int s, t;
  std::cin >> s >> t;
  s--, t--;
  std::cout << distance(adj, s, t);
  return 0;
}
