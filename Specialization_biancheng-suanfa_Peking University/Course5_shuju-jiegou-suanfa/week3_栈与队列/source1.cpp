#include <cctype>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  cin.get(); //使用getchar()可能会编译不通过
  while (n--) {
    char s[200];
    cin.getline(s, 200);
    for (int i = 0; s[i] != '\0'; i++) {
      if (isalpha(s[i])) {
        if (s[i] == 'z' || s[i] == 'Z')
          cout << char(s[i] - 25);
        else
          cout << char(s[i] + 1);
      } else
        cout << s[i];
    }
    cout << endl;
  }
  return 0;
}