#include <iostream>
#include <algorithm>
using namespace std;
struct D
{
	int start;
	int end;	
}d[100];

int cmp(D a,D b){
	return a.start<b.start;
}
int main() {
	int L,M;
	cin>>L>>M;
	for(int i=0;i<M;i++){
		cin>>d[i].start>>d[i].end;
	}
	sort(d,d+M,cmp);

	int mind =d[0].start;
	int maxend=d[0].end;
		int count = maxend-mind+1;
	for (int i = 1; i < M; ++i)
	{
		if(d[i].end<maxend)
			continue;
		else{
			if(d[i].start<maxend){
				count +=(d[i].end - maxend);
			}else{
				count +=(d[i].end - d[i].start+1);
			}
				maxend = d[i].end;
		}
	}
	cout<<L-count+1;
}