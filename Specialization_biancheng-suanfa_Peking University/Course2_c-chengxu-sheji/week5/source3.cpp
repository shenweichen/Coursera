#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;
char s[1005][85];
int num;


void PrintSingleWord(int k, int n) {
// kä¸ºç¬¬å‡ ä¸ªå•è¯,n ä¸ºè¯¥è¡Œå­—ç¬¦æ•°é‡?
if(k == num) return;
int len = strlen(s[k]);
if(n + len + 1 <= 80) { //n != 0è¦è€ƒè™‘ç©ºæ ¼ï¼ä½†å•è¯æ•°å°äº?0ï¼Œn ï¼?0ï¼Œåœ¨è¿™é‡Œæ— æ‰€è°“ã€?
//å¦‚æžœè¶³å¤Ÿå®¹çº³ï¼Œå°±è¾“å‡ºç©ºæ ¼ä¸Žå•è¯?
if (n != 0) cout << ' ';
cout << s[k];
if(n == 0) //è¿™é‡Œ n = 80 æ²¡æœ‰ç©ºæ ¼ï¼?
PrintSingleWord(k + 1, len);
else PrintSingleWord(k + 1, n + len + 1);

}else {
//ä¸å¤Ÿå®¹çº³ï¼Œæ¢è¡Œï¼
cout << endl;
cout << s[k];
PrintSingleWord(k + 1, len);
}
}


int main() {
cin >> num;
for (int i = 0; i < num; i++)
cin >> s[i];
PrintSingleWord(0, 0);
cout << endl;
return 0;
}


/*#include <iostream>
#include <algorithm>
using namespace std;



int main( ){
	int n;
	cin>>n;
	char str[100][41];
	cout<<n<<endl;
	for(int i=0;i<n;i++){
		cin>>str[i];
	}

	for(int i=0;i<n;i++)
		cout<<str[i]<<endl;
}*/