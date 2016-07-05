#include <iostream>
#include <algorithm>
using namespace std;
struct D
{
	int id;
	int before;
	int after;
	int rate;
}d[100];

bool cmp(D a,D b){
	return a.rate<b.rate;
}

int main() {
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		cin>>d[i].id>>d[i].before>>d[i].after;
		d[i].rate=d[i].after-d[i].before;
	}
	sort(d,d+n,cmp);

	int count ;
	for(int i=1;i<n-1;i++){
		if(d[i+1].rate-d[i].rate>d[i].rate-d[0].rate){
			count = i+1;
			break;
		}
	}


	cout<<n-count<<endl;
	for(int i=count;i<n;i++)
		cout<<d[i].id<<endl;
	cout<<count<<endl;
	for(int i=0;i<count;i++)
		cout<<d[i].id<<endl;


}