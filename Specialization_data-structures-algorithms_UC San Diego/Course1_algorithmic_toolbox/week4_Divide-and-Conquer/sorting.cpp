#include <cstdlib>
#include <iostream>
#include <vector>
using std::vector;
using std::swap;

int partition2(vector<int> &a, int l, int r) {
  int x = a[l];
  int j = l;
  for (int i = l + 1; i <= r; i++) {
    if (a[i] <= x) {
      j++;
      swap(a[i], a[j]);
    }
  }
  swap(a[l], a[j]);
  return j;
}
int partition3(vector<int> &a, int l, int r) {
  int x = a[l];
  int j = l;
  for (int i = l + 1; i <= r; i++) {
    if (a[i] <= x) {
      j++;
      swap(a[i], a[j]);
    }
  }
  swap(a[l], a[j]);
  // i-j为<=x的区间
  int i = l + 1, k = j;
  while (i < k) {
    while (a[i] != x && i < k)
      i++; //找到第一个等于x的元素
    while (a[k] == x && i < k)
      k--; //找到第一个不等于x的元素
    swap(a[i], a[k]);
    i++;
    k--;
  }
  return j;
}
void randomized_quick_sort(vector<int> &a, int l, int r) {
  if (l >= r) {
    return;
  }

  int k = l + rand() % (r - l + 1);
  swap(a[l], a[k]);

  int m = partition3(a, l, r);
  //连续的等于a[m]的元素不参与下一次排序
  int pos = m;
  while (a[m] == a[pos] && pos >= l)
    pos--;

  randomized_quick_sort(a, l, pos);
  randomized_quick_sort(a, m + 1, r);
}

int main() {
  int n;
  std::cin >> n;
  vector<int> a(n);
  for (size_t i = 0; i < a.size(); ++i) {
    std::cin >> a[i];
  }
  randomized_quick_sort(a, 0, a.size() - 1);
  for (size_t i = 0; i < a.size(); ++i) {
    std::cout << a[i] << ' ';
  }
  std::cout << std::endl;

  return 0;
}
