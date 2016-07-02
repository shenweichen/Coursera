#include <iostream>
#include <algorithm>
using namespace std;
const int MAXN = 20000;
int main() {
	int N;
	while (cin >> N) {
		if (N == 0)
			break;
		int a[MAXN] ;
		for (int i = 0; i<N; i++) {
			cin >> a[i];
		}
		sort(a,a+N);
		if(N%2==1){
			cout<<a[(N-1)/2]<<endl;
		}else{
			cout<<(a[N/2]+a[N/2-1])/2<<endl;
		}


	}



}