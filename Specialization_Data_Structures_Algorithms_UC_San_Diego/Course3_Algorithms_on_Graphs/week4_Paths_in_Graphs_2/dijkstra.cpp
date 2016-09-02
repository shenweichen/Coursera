#include <iostream>
#include <vector>
#include <queue>

using std::vector;
using std::queue;
using std::pair;
using std::priority_queue;


int distance(vector<vector<int> > &adj, vector<vector<int> > &cost, int s, int t) {
	//write your code her
	const int INF = 1000000000;
	vector<int> dis(adj.size(), INF);
	vector<bool> visited(adj.size(), 0);
	dis[s] = 0;

	for (size_t i = 0; i<adj.size(); i++) {
		int u = -1, MIN = INF;
		for (size_t j = 0; j<adj.size(); j++) {
			if (!visited[j] && dis[j]<MIN) {
				MIN = dis[j];
				u = j;
			}
		}
		if (u == -1)//........................
			break;
		visited[u] = 1;

		for (size_t j = 0; j<adj[u].size(); j++) {
			int next = adj[u][j];
			if (!visited[next] && dis[u] + cost[u][j]<dis[next]) {
				dis[next] = dis[u] + cost[u][j];
			}
		}

	}
	return dis[t] == INF ? -1 : dis[t];
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
	int s, t;
	std::cin >> s >> t;
	s--, t--;
	std::cout << distance(adj, cost, s, t);
	return 0;
}
