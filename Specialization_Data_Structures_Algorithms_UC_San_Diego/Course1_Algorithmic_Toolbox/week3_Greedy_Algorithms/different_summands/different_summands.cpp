#include <iostream>
#include <vector>

using std::vector;

vector<int> optimal_summands(int n) {
  vector<int> summands;
  int sum = 0,k=n,cur=1;
  while(sum!=n){
    if(k>2*cur){
      sum+=cur;
      summands.push_back(cur);
      k-=cur;
      cur++;
    }else{
      sum+=k;
      summands.push_back(k);
      break;
    }
  }
  return summands;
}

int main() {
  int n;
  std::cin >> n;
  vector<int> summands = optimal_summands(n);
  std::cout << summands.size() << '\n';
  for (size_t i = 0; i < summands.size(); ++i) {
    std::cout << summands[i] << ' ';
  }
  return 0;
}
