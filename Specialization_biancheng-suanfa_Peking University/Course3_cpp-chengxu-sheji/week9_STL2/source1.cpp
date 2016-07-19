/*#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
struct Point {
    int x;
    int y;
    Point(int x_,int y_):x(x_),y(y_) { }
};
bool operator < ( const Point & p1, const Point & p2)
{
    if( p1.y < p2.y )
        return true;
    else if( p1.y == p2.y )
        return p1.x < p2.x;
    else
        return false;
}
int main()
{
    int t;
    int x,y;
    cin >> t;
    vector<Point> v;
    while( t -- ) {
        cin >> x >> y;
        v.push_back(Point(x,y));
    }
    vector<Point>::iterator i,j;
    int nTotalNum = 0;*/

    sort(v.begin(),v.end());
    for( i = v.begin(); i < v.end() - 1;i ++ )
        for(j=i+1; j<v.end(); j++) {
            if(binary_search(v.begin(),v.end(),Point( j->x, i->y)) &&
                binary_search(v.begin(),v.end(),Point( i->x, j->y))&&
                i->x!=j->x &&
              i->y!=j->y )
                nTotalNum ++;
        }
    cout << nTotalNum/2;

/*return 0;
}*/