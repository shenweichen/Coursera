
#include <iostream>
#include <vector>
using namespace std;
int guess(int n, vector<vector<int> > a, vector<vector<int> > paint) {
	int count = 0;
	for (int i = 1; i < n ; i++) {
		for (int j = 1; j < n + 1; j++) {		
			paint[i+1][j]=(a[i][j]+paint[i][j]+paint[i-1][j]+paint[i][j-1]+paint[i][j+1])%2;
            //0 + 奇数 说明 a[i][j]是1 这时候 不需要
			}
		}
    		//判断最后一行是否全为1？
		for (int j = 1; j<n + 1; j++) {
			if((a[n][j]+paint[n][j]+paint[n-1][j]+paint[n][j-1]+paint[n][j+1])%2==1)
				return 1001;
		}
    //算出需要的步数
    for (int i=1; i<=n; i++) {
        for (int j=1; j<=n; j++) {
            if (paint[i][j]==1) {
                count++;
            }
        }
    }


		return count;
	}

	int enumerate(const int & n, vector<vector<int> > a, vector<vector<int> > paint) {
		int minsteps = 1001;
		int steps;
		int c ;
        int state = 0;
		while (paint[1][n + 1] != 1) {
			steps = guess(n, a, paint);
			if (steps<minsteps)
				minsteps = steps;
			paint[1][1]++;
            c=1;
			while (paint[1][c]>1) {
				paint[1][c] = 0;
				paint[1][c + 1]++;
				c++;
			}
		}
		return minsteps;
	}


	int main() {
		int t;
		cin >> t;
		while (t--) {
			int n;
			cin >> n;
			vector<vector<int> > a(n + 2, vector<int>(n + 2));
			vector<vector<int> > paint(n + 2, vector<int>(n + 2, 0));
			for (int i = 1; i < n + 1; i++) {
				for (int j = 1; j < n + 1; j++) {
					char temp;
					cin >> temp;
					a[i][j] = temp == 'y' ? 0 : 1;
				}
			}
			int ans = enumerate(n, a, paint);
            if(ans==1001)
                cout<<"inf"<<endl;
            else
			cout << ans << endl;
		}
		return 0;
	}
