#include <algorithm>
#include <iostream>
#include <vector>

using std::vector;
using std::cin;
using std::cout;
using std::swap;
using std::pair;
using std::make_pair;

class HeapBuilder {
private:
  vector<int> data_;
  vector<pair<int, int>> swaps_;

  void WriteResponse() const {
    cout << swaps_.size() << "\n";
    for (int i = 0; i < swaps_.size(); ++i) {
      cout << swaps_[i].first << " " << swaps_[i].second << "\n";
    }
  }

  void ReadData() {
    int n;
    cin >> n;
    data_.resize(n);
    for (int i = 0; i < n; ++i)
      cin >> data_[i];
  }
  void SiftUp(int i) {
    while (i > 0 && data_[(i - 1) >> 1] > data_[i]) {
      swap(data_[(i - 1) >> 1], data_[i]);
      swaps_.push_back(make_pair((i - 1) >> 1, i));
      i = (i - 1) >> 1;
    }
  }
  void SiftDown(int i) {
    int minIndex = i; //建立小根堆，记录当前位置为最小
    int l = 2 * i + 1, r = 2 * i + 2; //与左右孩子分别比较，记录三者最小值
    if (l < data_.size() && data_[l] < data_[minIndex])
      minIndex = l;
    if (r < data_.size() && data_[r] < data_[minIndex])
      minIndex = r;
    if (minIndex != i) { //当前位置不是最小值，交换
      swap(data_[i], data_[minIndex]);
      swaps_.push_back(make_pair(i, minIndex));
      ShiftDown(minIndex); //对交换后的孩子结点依次向下调整
    }
  }
  void GenerateSwaps() {
    swaps_.clear();
    // The following naive implementation just sorts
    // the given sequence using selection sort algorithm
    // and saves the resulting sequence of swaps.
    // This turns the given array into a heap,
    // but in the worst case gives a quadratic number of swaps.
    //
    // TODO: replace by a more efficient implementation
    for (int i = (data_.size() - 2) >> 1; i >= 0; i--) //建堆，整体向上调整
      SiftDown(i);

    /* for (int i = 0; i < data_.size(); ++i)
       for (int j = i + 1; j < data_.size(); ++j) {
         if (data_[i] > data_[j]) {
           swap(data_[i], data_[j]);
           swaps_.push_back(make_pair(i, j));
         }
       }*/
  }

public:
  void Solve() {
    ReadData();
    GenerateSwaps();
    WriteResponse();
  }
};

int main() {
  std::ios_base::sync_with_stdio(false);
  HeapBuilder heap_builder;
  heap_builder.Solve();
  return 0;
}
