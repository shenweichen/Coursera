#include <iostream>
#include <string>
#include <vector>


using std::string;
using std::vector;
using std::cin;

struct Query {
  string type, name;
  int number;
};

vector<Query> read_queries() {
  int n;
  cin >> n;
  vector<Query> queries(n);
  for (int i = 0; i < n; ++i) {
    cin >> queries[i].type;
    if (queries[i].type == "add")
      cin >> queries[i].number >> queries[i].name;
    else
      cin >> queries[i].number;
  }
  return queries;
}

void write_responses(const vector<string> &result) {
  for (size_t i = 0; i < result.size(); ++i)
    std::cout << result[i] << "\n";
}

vector<string> process_queries(const vector<Query> &queries) {
  vector<string> result;
  // Keep list of all existing (i.e. not deleted yet) contacts.
  vector<vector<Query> > contacts(100000);
  for (size_t i = 0; i < queries.size(); ++i) {
    int hash = queries[i].number % 100000;
    if (queries[i].type == "add") {
      bool was_founded = false;

      // if we already have contact with such number,
      // we should rewrite contact's name
      for (size_t j = 0; j < contacts[hash].size(); ++j)
        if (contacts[hash][j].number == queries[i].number) {
          contacts[hash][j].name = queries[i].name;
          was_founded = true;
          break;
        }
      // otherwise, just add it
      if (!was_founded)
        contacts[hash].push_back(queries[i]);
    } else if (queries[i].type == "del") {

      for (size_t j = 0; j < contacts[hash].size(); ++j)
        if (contacts[hash][j].number == queries[i].number) {
          contacts[hash].erase(contacts[hash].begin() + j);
          break;
        }
    } else {
      string response = "not found";
      for (size_t j = 0; j < contacts[hash].size(); ++j)
        if (contacts[hash][j].number == queries[i].number) {
          response = contacts[hash][j].name;
          break;
        }
      result.push_back(response);
    }
  }
  return result;
}

int main() {
  write_responses(process_queries(read_queries()));
  return 0;
}
