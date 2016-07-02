#include <iostream>
#include <cmath>
using namespace std;
int main() {
	int n, x, y;
	cin >> n >> x >> y;
	int num = ceil((double)y / x);
	int ans = n - num > 0 ? n - num : 0;
		cout << ans<< endl;
	return 0;
}
