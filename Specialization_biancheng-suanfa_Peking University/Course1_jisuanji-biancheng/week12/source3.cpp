#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
using namespace std;
struct Loc
{
	double x;
	double y;
}loc[1000];
int main() {
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		cin>>loc[i].x>>loc[i].y;
	}
	double dis = -1.0;
	for(int i=0;i<n-1;i++){
		for(int j=i+1;j<n;j++){
			double temp = (loc[i].x-loc[j].x)*(loc[i].x-loc[j].x)+(loc[i].y-loc[j].y)*(loc[i].y-loc[j].y);
			dis = max(temp,dis);
		}
	}
	cout << fixed << setprecision(4) << sqrt(dis) << endl;
}