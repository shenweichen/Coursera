#include <iostream>
#include <map>
#include <iomanip>
using namespace std;

bool isrun(int year){
	if(year %4==0&&year%100!=0)
		return true;
	if(year%400==0&&year%1000==0)
		return true;
	return false;
}

int month_day[2][13]={{0,31,28,31,30,31,30,31,31,30,31,30,31},{0,31,29,31,30,31,30,31,31,30,31,30,31}};
int main() {
int y,m,d;
char delimiter;
	cin>>y>>delimiter>>m>>delimiter>>d;
	d=d+1;
	bool runnian = isrun(y);
	if(d>month_day[runnian][m]){
		d=d-month_day[runnian][m];
		m=m+1;
		if(m>12){
			m=1;
			y++;
		}
	}
	cout<<y<<"-"<<setfill('0')<<setw(2)<<m<<"-"<<setfill('0')<<setw(2)<<d;


}