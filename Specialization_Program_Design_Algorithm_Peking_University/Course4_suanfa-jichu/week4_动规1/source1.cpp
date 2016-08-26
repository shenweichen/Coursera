#include <iostream>
#include <map>
using namespace std;
void input(map<int, int> &a, int &len) {
	int temp;
	for (int i = 0; i < len; i++) {
		cin >> temp;
		if (a.find(temp) != a.end())
			a[temp]++;
		else
			a[temp] = 1;
	}
}
void count(map<int, int> &a, map<int, int> &b, int &s) {
	int count = 0;
	for (map<int, int>::iterator it = a.begin(); it != a.end(); it++) {
		if (b.find(s - it->first) != b.end()) {
			count += it->second*b[s - it->first];
		}
	}
	cout << count << endl;
}
int main() {
	int n;
	cin >> n;
	while (n--) {
		map<int, int> a, b;
		int len, s;
		cin >> s;
		cin >> len;
		input(a, len);
		cin >> len;
		input(b, len);
		count(a, b, s);
	}
}