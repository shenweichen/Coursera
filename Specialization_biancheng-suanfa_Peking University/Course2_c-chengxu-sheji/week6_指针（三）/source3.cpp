#include<stdio.h>
using namespace std;

int main() {
    int m, n, i, j;
    scanf("%d %d", &m, &n);
    int a[m][n];
    for (i=0;i<m;i++) {
        for (j=0;j<n;j++) {
            scanf("%d", &a[i][j]);
        }
    }
    for (i=0;i<m;i++) {
        for (j=0;j<n;j++) {
            bool shan = true;
            
            if (shan && i-1>=0) {
                if (a[i][j]<a[i-1][j]) {
                    shan = false;
                }
            }
            
            if (shan && i+1<m) {
                if (a[i][j]<a[i+1][j]) {
                    shan = false;
                }
            }
            
            if (shan && j-1>=0) {
                if (a[i][j]<a[i][j-1]) {
                    shan = false;
                }
            }
            
            if (shan && j+1<n) {
                if (a[i][j]<a[i][j+1]) {
                    shan = false;
                }
            }
            
            if (shan) {
                printf("%d %d\n", i, j);
            }
        }
    }
    return 0;
}