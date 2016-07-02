#include <iostream>
#include <cmath>
using namespace std;
int main() {
    int n;
    int odd=-1;
    int even=100;
    for(int i=0;i<6;i++){
        cin>>n;
        if(n%2==0)
            even = n<even?n:even;
        else
            odd = n>odd?n:odd;
    }
    cout<<abs(odd-even)<<endl;
}
