#include <iostream>
#include <vector>
#include <queue>

using std::vector;
using std::queue;

int bipartite(vector<vector<int> > &adj) {
  //write your code here
  vector<int> visited(adj.size(),0);//0未访问，1/-1代表颜色
  queue<int> q;
  q.push(0);
  visited[0]=1;
  while(!q.empty()){
    int cur = q.front();
    q.pop();
    for(size_t i =0;i<adj[cur].size();i++){
      int next = adj[cur][i];
      if(visited[next]==0){
        q.push(next);
        visited[next]=-visited[cur];
      }else if(visited[next]==visited[cur])
      return 0;
    }
  }
  return 1;
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
  std::cout << bipartite(adj);
  return 0;
}
