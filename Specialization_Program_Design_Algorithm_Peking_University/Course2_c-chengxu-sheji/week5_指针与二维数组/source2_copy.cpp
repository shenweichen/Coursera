#include<stdio.h>
int main()
{
 int a[100][100];
 int i,j,m,n,p;
 scanf("%d%d", &m, &n);
 for(i=0;i<m;i++)
 {
  for(j=0;j<n;j++)
  {
   scanf("%d", &a[i][j]);
  }
 }
 for(p=0;p<m+n-1;p++)
 {
  for(i=0;i<=p;i++)
  {
   if(i<m&&p-i<n)
   {
    printf("%d\n", a[i][p-i]);
   }
   else
    continue;
  }
 }
 return 0;
}

/*#include <iostream>
#include <algorithm>
using namespace std;

int main( ){
		int m,n;
		cin>>m>>n;
		int a[100][100];
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++)
				cin>>a[i][j];
		}
		int last_i=0,last_j=0;
		for(int i=0,j=0,count=0;count<m*n;count++){
			cout<<a[i][j]<<endl;
			if(i+1<m&&j-1>=0){
				i++;
				j--;
			}
			else if(count<(double)(m*n)/2){
				i=last_i;
				j=last_j+1;
				last_i=i;
				last_j=j;
				//cout<<last_i<<" "<<last_j<<endl;
			}
			else{
				i=last_i+1;
				j=last_j;
				last_i=i;
				last_j=j;
				//cout<<last_i<<" "<<last_j<<endl;
			}

		}

		

	
}*/