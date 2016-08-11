#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace std;
int res, n;
int flag[8];
void DFS(int row, const vector<vector<int> > &a, int k) {
  if (k == 0) {
    res++;
    return;
  }
  if (row == n)
    return;
  for (int j = 0; j < n; j++) {
    if (flag[j] == 0 && a[row][j] == 0) {
      flag[j] = 1;
      DFS(row + 1, a, k - 1);
	  flag[j]=0;
    } 
      
  }
  DFS(row + 1, a, k);
}
int main() {
  int k;

  while (cin >> n >> k) {
    if (n == -1 && k == -1)
      break;
    vector<vector<int> > a(n, vector<int>(n, 0));
    fill(flag, flag + 8, 0);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        char temp;
        cin >> temp;
        if (temp == '.')
          a[i][j] = 1;
      }
    }
    res = 0;
    DFS(0, a, k);
    cout << res << endl;
  }
  return 0;
}