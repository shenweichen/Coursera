#include <iostream>
#include <bitset>
#include <iomanip>
using namespace std;

int main(){
	int  t;
	cin>>t;
	while(t--){
		int n,i,j;
		cin>>n>>i>>j;
		bitset<40> bst(n);
		bitset<40> k;
		k[i]=bst[i];
		k[j]=~bst[j];
		for(int pos = i+1;pos<j;pos++)
			k.set(pos,1);
		n = k.to_ulong();
		cout<<hex<<n<<endl;
	}

}