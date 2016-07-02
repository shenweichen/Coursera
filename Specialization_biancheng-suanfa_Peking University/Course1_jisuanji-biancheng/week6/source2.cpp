#include <iostream>
#include <cmath>
using namespace std;
int main() {
    const double pi = 3.14159;
    int h, r;
    cin >> h >> r;
    double V = pi*r*r*h;
    cout << ceil(20000 / V) << endl;
    return 0;
}
