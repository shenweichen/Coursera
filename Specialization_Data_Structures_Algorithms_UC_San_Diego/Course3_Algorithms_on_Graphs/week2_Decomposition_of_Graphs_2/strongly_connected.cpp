#include <algorithm>
#include <iostream>
#include <vector>

using std::vector;
using std::pair;

void DFS(int start,vector<vector<int> > &GR,vector<bool> &visited,vector<int> &order){
visited[start]=true;
for(size_t i=0;i<GR[start].size();i++){
  if(!visited[GR[start][i]])
  DFS(GR[start][i],GR,visited,order);
}
order.push_back(start);
}
void explore(int start,vector<vector<int> > &GR,vector<bool> &visited){
visited[start]=true;
for(size_t i=0;i<GR[start].size();i++){
  if(!visited[GR[start][i]])
  explore(GR[start][i],GR,visited);
}

}

int number_of_strongly_connected_components(vector<vector<int> > adj) {
  int result = 0;
  //write your code here
  vector<vector<int> > GR(adj.size());//构造GR
  for(size_t i =0;i<adj.size();i++){
    for(size_t j=0;j<adj[i].size();j++){
      GR[adj[i][j]].push_back(i);
    }
  }
  vector<bool> visited(adj.size());
  vector<int> order;
for(size_t i=0;i<GR.size();i++){//获得reverse post order
  if(!visited[i])
  DFS(i,GR,visited,order);
}
reverse(order.begin(),order.end());
fill(visited.begin(),visited.end(),0);
for(size_t i =0;i<order.size();i++){//获取SCCs
  if(!visited[order[i]]){
  explore(order[i],adj,visited);
  result++;
  }
}

  return result;
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
  std::cout << number_of_strongly_connected_components(adj);
  return 0;
}
