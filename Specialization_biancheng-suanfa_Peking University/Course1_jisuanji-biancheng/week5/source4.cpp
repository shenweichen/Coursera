# include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        int x, ans = 0;
        cin >> x;
        while (x > 0) {
            ans += x % 2;
            x /= 2;
        }
        cout << ans << endl;
    }
    return 0;
}