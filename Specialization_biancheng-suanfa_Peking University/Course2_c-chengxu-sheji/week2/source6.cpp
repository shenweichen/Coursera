#include <iostream>
#include <algorithm>
using namespace std;

char a[100][100];
int n;
void flu(int i,int j){
	if(i-1>=0&&a[i-1][j]=='.')
		a[i-1][j]='$';
	if(i+1<=n-1&&a[i+1][j]=='.')
		a[i+1][j]='$';
	if(j-1>=0&&a[i][j-1]=='.')
		a[i][j-1]='$';
	if(j+1<=n-1&&a[i][j+1]=='.')
		a[i][j+1]='$';
}

int main() {

	cin>>n;
	for(int i=0;i<n;i++){
		for (int j=0;j<n;j++)
			cin>>a[i][j];
		cin.get();
	}
	int m;
	cin>>m;

	for(int k=1;k<m;k++){
	for(int i=0;i<n;i++){
		for (int j=0;j<n;j++)
			{
				if(a[i][j]=='@')
					flu(i,j);

			}
	}
for(int i=0;i<n;i++){
		for (int j=0;j<n;j++)
			{
				if(a[i][j]=='$')
				a[i][j]='@';

			}
		
	}

	}
	int sum = 0;
	for(int i=0;i<n;i++){
		for (int j=0;j<n;j++)
			{
				if(a[i][j]=='@')
					sum++;

			}
	}
cout<<sum<<endl;

	
}