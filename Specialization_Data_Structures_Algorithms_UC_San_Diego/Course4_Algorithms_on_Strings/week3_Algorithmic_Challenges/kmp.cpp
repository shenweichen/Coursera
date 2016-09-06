#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::string;
using std::vector;

// Find all occurrences of the pattern in the text and return a
// vector with all positions in the text (starting from 0) where 
// the pattern starts in the text.
vector<int> find_pattern(const string& pattern, const string& text) {
  vector<int> result;
  // Implement this function yourself
  string s = pattern+"$"+text;
  vector<int> next(s.size());
  next[0]=0;
  int border = 0;
  for(size_t i =1;i<s.size();i++){//compute prefix function
    while(border>0&&s[i]!=s[border])
    border = next[border-1];
    if(s[i]==s[border])
    border++;
    else
    border = 0;
    next[i]=border;
  }

  for(size_t i =pattern.size()+1;i<s.size();i++){//KMP
  if(next[i]==pattern.size())
  result.push_back(i-2*pattern.size());
  }

  return result;
}

int main() {
  string pattern, text;
  cin >> pattern;
  cin >> text;
  vector<int> result = find_pattern(pattern, text);
  for (int i = 0; i < result.size(); ++i) {
    printf("%d ", result[i]);
  }
  printf("\n");
  return 0;
}
