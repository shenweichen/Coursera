#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

using std::vector;
using std::cin;
using std::cout;
using std::priority_queue;

class JobQueue {
private:
  int num_workers_;
  vector<int> jobs_;

  vector<int> assigned_workers_;
  vector<long long> start_times_;
  struct thread {
    int id;
    long long next_free_time;
    thread(int id, long long t = 0) : id(id), next_free_time(t){};
    void addtime(long long t) { next_free_time += t; }
    friend bool operator<(const thread &b, const thread &a) {
      if (b.next_free_time != a.next_free_time)
        return b.next_free_time > a.next_free_time;
      else
        return b.id > a.id;
    }
  };
  void WriteResponse() const {
    for (int i = 0; i < jobs_.size(); ++i) {
      cout << assigned_workers_[i] << " " << start_times_[i] << "\n";
    }
  }

  void ReadData() {
    int m;
    cin >> num_workers_ >> m;
    jobs_.resize(m);
    for (int i = 0; i < m; ++i)
      cin >> jobs_[i];
  }

  void AssignJobs() {
    // TODO: replace this code with a faster algorithm.
    assigned_workers_.resize(jobs_.size());
    start_times_.resize(jobs_.size());
    // vector<long long> next_free_time(num_workers_, 0);
    priority_queue<thread, vector<thread>> q; //使用优先队列
    for (int i = 0; i < num_workers_; i++) {
      q.push(thread(i));
    }
    for (int i = 0; i < jobs_.size(); ++i) {
      int duration = jobs_[i];
      thread td = q.top();
      q.pop();
      assigned_workers_[i] = td.id;
      start_times_[i] = td.next_free_time; // next_free_time[next_worker];
      td.next_free_time += duration;
      q.push(td);
    }
  }

public:
  void Solve() {
    ReadData();
    AssignJobs();
    WriteResponse();
  }
};

int main() {
  std::ios_base::sync_with_stdio(false);
  JobQueue job_queue;
  job_queue.Solve();
  return 0;
}
