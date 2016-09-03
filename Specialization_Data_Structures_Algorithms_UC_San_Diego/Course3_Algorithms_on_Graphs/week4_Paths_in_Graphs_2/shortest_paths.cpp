#include <iostream>
#include <limits>
#include <vector>
#include <queue>

using std::vector;
using std::queue;
using std::pair;
using std::priority_queue;

void shortest_paths(vector<vector<int> > &adj, vector<vector<int> > &cost, int s, vector<long long> &distance, vector<int> &reachable, vector<int> &shortest) {
  //write your code here
  distance[s]=0;
  reachable[s]=1;//自身可达
  for(size_t k =1;k<adj.size();k++){//循环V-1次计算最短路径
    for(size_t i =0;i<adj.size();i++){
      for(size_t j =0;j<adj[i].size();j++){
        int next = adj[i][j];
        if(distance[i]!=std::numeric_limits<long long>::max()&&distance[i]+cost[i][j]<distance[next]){//这里判定distance[i]是否为极大，防止加法溢出
          distance[next]=distance[i]+cost[i][j];
          reachable[next] = 1;
        }
      }
    }
  }
//vector<bool> inq(adj.size(),0);//标记负权环中的点
queue<int> q;
 for(size_t i =0;i<adj.size();i++){
      for(size_t j =0;j<adj[i].size();j++){
        int next = adj[i][j];
        if(distance[i]!=std::numeric_limits<long long>::max()&&distance[i]+cost[i][j]<distance[next]){//第V次迭代仍能松弛的点一定在负权环中
        //第一次没过，因为这里也要判定distance[i]是否为极大，防止加法溢出
          distance[next]=distance[i]+cost[i][j];
          if(shortest[next]){
            q.push(next);
            shortest[next]=0;   //标记0最短路不存在
            }
        }
      }
    }

    while(!q.empty()){//负权环上的点可达的点均为可以无限套利的点。不需要判定这些点是否源点可达，因为源点可达的点在第一次循环已经标注完成。
      int cur = q.front();
      q.pop();
      for(size_t i =0;i<adj[cur].size();i++){
        int next = adj[cur][i];
        if(shortest[next]){
        q.push(next);
        shortest[next]=0;
        }
      }
    }


}

int main() {
  int n, m, s;
  std::cin >> n >> m;
  vector<vector<int> > adj(n, vector<int>());
  vector<vector<int> > cost(n, vector<int>());
  for (int i = 0; i < m; i++) {
    int x, y, w;
    std::cin >> x >> y >> w;
    adj[x - 1].push_back(y - 1);
    cost[x - 1].push_back(w);
  }
  std::cin >> s;
  s--;
  vector<long long> distance(n, std::numeric_limits<long long>::max());
  vector<int> reachable(n, 0);
  vector<int> shortest(n, 1);
  shortest_paths(adj, cost, s, distance, reachable, shortest);
  for (int i = 0; i < n; i++) {
    if (!reachable[i]) {
      std::cout << "*\n";
    } else if (!shortest[i]) {
      std::cout << "-\n";
    } else {
      std::cout << distance[i] << "\n";
    }
  }
  return 0;
}
