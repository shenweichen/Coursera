#include<iostream>
#include<string>
#include<vector>
#include<cstdio>
#include<map>
using namespace std;

const int inf = 21 * 50;
const int maxn = 30;
struct Edge {
	int dis;
	int next;
	Edge(int dis, int next) :dis(dis), next(next) {

	}
};
vector<vector<Edge> > Adj(maxn);

vector<int> Dijkstra(int s,int n,vector<int> &adjdis) {
	vector<bool> collected(n);
	vector<int> dis(n, inf);
	vector<int> path(n, -1);
	dis[s] = 0;
	collected[s] = true;
	for (int i = 0; i < Adj[s].size(); i++) {
		Edge e = Adj[s][i];
		if (e.dis < dis[e.next]) {
			dis[e.next] = e.dis;
			path[e.next] = s;
			adjdis[e.next] = e.dis;
		}
	}

	while (1) {
		int v = -1;
		int mindis = inf;
		for (int i = 0; i < n; i++) {
			if (collected[i] == false && dis[i] < mindis) {
				mindis = dis[i];
					v = i;
			}
		}
		if (v == -1)
			break;
		collected[v] = true;
		for (int i = 0; i < Adj[v].size(); i++) {
			Edge e = Adj[v][i];
			if (dis[v]+e.dis<dis[e.next]) {
				dis[e.next] = dis[v]+e.dis;
				path[e.next] = v;
				adjdis[e.next] = e.dis;
			}
		}
	}
	
	return path;
}

void getPath(vector<int> &ans,int s,vector<int> const & path ) {
	if (path[s] != -1)
		getPath(ans,path[s], path);
	ans.push_back(s);
}
void printAns(vector<int> const & ans, vector<int> const & adjdis,map<int, string> int2str) {
	bool first = true;
	for (int i = 0; i < ans.size(); i++) {
		if (first) {
			cout << int2str[ans[i]];
			first = false;
		}
		else {
			cout << "->(" << adjdis[ans[i]] << ")->" << int2str[ans[i]];
		}
	}
	//cout << endl;
}
int main() {
	int N,M,R;
	map<string, int> str2int;
	map<int, string> int2str;
	cin >> N;
	for (int i = 0; i < N; i++) {
		string place;
		cin >> place;
		str2int[place] = i;
		int2str[i] = place;
	}

	cin >> M;
	for (int i = 0; i < M; i++) {
		string sa, sb;
		int weight;
		cin >> sa >> sb >> weight;
		int a = str2int[sa], b = str2int[sb];
		Adj[a].push_back(Edge(weight, b));
		Adj[b].push_back(Edge(weight, a));
	}

	cin >> R;
	for (int i = 0; i < R; i++) {
		string sa, sb;
		cin >> sa >> sb ;
		int a = str2int[sa], b = str2int[sb];
		vector<int> adjdis(N, -1);
		vector<int> path = Dijkstra(a, N,adjdis);
		vector<int> ans;
		getPath(ans,b, path);
		printAns(ans, adjdis, int2str);
	}
	return 0;
}