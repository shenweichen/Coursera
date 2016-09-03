#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
using namespace std;
vector<vector<int> > adj;
void dfs(int s, vector<bool> &visited,vector<int> &result) {
	visited[s] = true;
	sort(adj[s].begin(), adj[s].end());
	for (int i = adj[s].size() - 1; i >= 0; i--) {
		int next = adj[s][i];
		if (!visited[next])
			dfs(next, visited,result);
	}
	result.push_back(s);
}

vector<int> DFSTraverse(int n){
  vector<int> result;
  	vector<bool> visited(n, 0);
  	for (int i = n - 1; i >= 0; i--) {
		if (!visited[i])
			dfs(i,visited,result);
	}
  reverse(result.begin(),result.end());
  return result;
}
class MyCmp
{
public:
  bool operator()(const int &a,const int &b){
    return a>b;
  }

};

vector<int> toposort(vector<int> inDegree){//拓扑排序
  priority_queue<int,vector<int> ,MyCmp >  q;
  vector<int> result;
for(size_t  i=0;i<inDegree.size();i++){
  if(inDegree[i]==0)
  q.push(i);
}
while(!q.empty()){
int cur = q.top();
q.pop();
result.push_back(cur);
for(size_t i =0;i<adj[cur].size();i++){
  int next = adj[cur][i];
  inDegree[next]--;
  if(inDegree[next]==0)
    q.push(next);
}
}
return result;
};
int main() {
	int n, m;
	cin >> n >> m;
  adj.resize(n);
  vector<int> inDegree(n,0);
	while (m--) {
		int s, e;
		cin >> s >> e;
		adj[s-1].push_back(e-1);
    inDegree[e-1]++;
	}
  //vector<int> result = DFSTraverse(n);  DFS后序遍历逆序
  vector<int> result = toposort(inDegree);//拓扑排序
	for (int i = 0; i < n; i++) {
		cout << "v" << result[i] + 1;
		if(i<n-1)
		cout<< " ";
	}
	return 0;

}