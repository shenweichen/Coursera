#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

string BWT(const string &text) {
  string result = "";

  // write your code here
  vector<string> BWTMatrix;
  BWTMatrix.push_back(text);
  for (size_t i = 1; i < text.size(); i++) {
    string temp;
    temp.resize(text.size()); //这里不重新分配的话之后的rotate_copy会出错
    std::rotate_copy(text.begin(), text.begin() + i, text.end(),
                     temp.begin()); //循环左移
    BWTMatrix.push_back(temp);
  }
  std::sort(BWTMatrix.begin(), BWTMatrix.end());
  for (size_t i = 0; i < BWTMatrix.size(); i++)
    result += BWTMatrix[i][text.size() - 1];
  return result;
}

int main() {
  string text;
  cin >> text;
  cout << BWT(text) << endl;
  return 0;
}