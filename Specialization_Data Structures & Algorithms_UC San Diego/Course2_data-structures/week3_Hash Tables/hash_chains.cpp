#include <algorithm>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::cin;
using std::list;

struct Query {
  string type, s;
  size_t ind;
};

class QueryProcessor {
  int bucket_count;
  // store all strings in one vector
  // vector<string> elems;
  vector<list<string>> elems;
  size_t hash_func(const string &s) const {
    static const size_t multiplier = 263;
    static const size_t prime = 1000000007;
    unsigned long long hash = 0;
    for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i)
      hash = (hash * multiplier + s[i]) % prime;
    return hash % bucket_count;
  }

public:
  explicit QueryProcessor(int bucket_count) : bucket_count(bucket_count) {
    elems.resize(bucket_count);
  }

  Query readQuery() const {
    Query query;
    cin >> query.type;
    if (query.type != "check")
      cin >> query.s;
    else
      cin >> query.ind;
    return query;
  }

  void writeSearchResult(bool was_found) const {
    std::cout << (was_found ? "yes\n" : "no\n");
  }

  void processQuery(const Query &query) {
    if (query.type == "check") {
      list<string> &li = elems[query.ind];
      for (list<string>::iterator it = li.begin(); it != li.end(); it++) {
        std::cout << *it << " ";
      }
      // use reverse order, because we append strings to the end
      /* for (int i = static_cast<int>(elems.size()) - 1; i >= 0; --i)
      if (hash_func(elems[i]) == query.ind)
      std::cout << elems[i] << " ";*/
      std::cout << "\n";
    } else {
      // vector<string>::iterator it = std::find(elems.begin(), elems.end(),
      // query.s);
      list<string> &li = elems[hash_func(query.s)];
      list<string>::iterator it;
      if (query.type == "find") {
        bool was_found = false;
        for (it = li.begin(); it != li.end(); it++) {
          if (*it == query.s) {
            was_found = true;
            break;
          }
        }
        writeSearchResult(was_found);
      } else if (query.type == "add") {

        for (it = li.begin(); it != li.end(); it++) {
          if (*it == query.s)
            break;
        }
        if (it == li.end())
          li.push_front(query.s);
      } else if (query.type == "del") {
        for (it = li.begin(); it != li.end(); it++) {
          if (*it == query.s) {
            li.erase(it);
            break;
          }
        }
      }
    }
  }

  void processQueries() {
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i)
      processQuery(readQuery());
  }
};

int main() {
  std::ios_base::sync_with_stdio(false);
  int bucket_count;
  cin >> bucket_count;
  QueryProcessor proc(bucket_count);
  proc.processQueries();
  return 0;
}
