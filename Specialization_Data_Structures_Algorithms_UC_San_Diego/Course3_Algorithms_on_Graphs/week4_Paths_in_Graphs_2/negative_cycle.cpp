#include <iostream>
#include <vector>

using std::vector;

int negative_cycle(vector<vector<int> > &adj, vector<vector<int> > &cost) {
  //write your code here
  const int INF = 100000000;
  vector<int> dis(adj.size(),INF);
  dis[0]=0;
  for(size_t k =1;k<adj.size();k++){//重复V-1次
for(size_t i =0;i<adj.size();i++){//对所有的边进行松弛
  for(size_t j =0;j<adj[i].size();j++){
    int next = adj[i][j];
    if(dis[i]+cost[i][j]<dis[next])
      dis[next]=dis[i]+cost[i][j];
  }
}
}

for(size_t i =0;i<adj.size();i++){//判断是否存在负权环
  for(size_t j =0;j<adj[i].size();j++){
    int next = adj[i][j];
    if(dis[i]+cost[i][j]<dis[next])
      return 1;
}}

  return 0;
}

int main() {
  int n, m;
  std::cin >> n >> m;
  vector<vector<int> > adj(n, vector<int>());
  vector<vector<int> > cost(n, vector<int>());
  for (int i = 0; i < m; i++) {
    int x, y, w;
    std::cin >> x >> y >> w;
    adj[x - 1].push_back(y - 1);
    cost[x - 1].push_back(w);
  }
  std::cout << negative_cycle(adj, cost);
  return 0;
}
