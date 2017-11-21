#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

int main() {
	int N, M;
	vector<int> w;
	vector<int> d;
	cin >> N >> M;
	for (int i = 0; i < N; i++) {
		int temp_w, temp_d;
		cin >> temp_w >> temp_d;
		w.push_back(temp_w);
		d.push_back(temp_d);
	}
	vector<int> f(M+1,0);
	for (int i = 0; i < N; i++) {
		for (int v = M; v >= w[i]; v--) {
			f[v] = max(f[v], f[v - w[i]] + d[i]);
		}
	}
	cout << f[M];
	return 0;

}
