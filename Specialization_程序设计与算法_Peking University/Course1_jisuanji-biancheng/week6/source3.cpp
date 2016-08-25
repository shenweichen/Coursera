#include <iostream>
#include <cmath>
using namespace std;
int main() {
    int n;
    cin >> n;
    int max = -1;
    for (int i = 0; i < n; i++) {
        int score;
        cin >> score;
        max = score > max ? score : max;
    }
    cout << max << endl;
    return 0;
}
