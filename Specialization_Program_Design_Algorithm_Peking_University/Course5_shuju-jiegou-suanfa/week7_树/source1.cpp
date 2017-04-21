#include<iostream>
#include<string>
#include<vector>
#include<cstdio>
#include <ctype.h>


using namespace std;
const int maxn = 500;
const int inf = 501 * maxn;

struct Node {
	char value;
	vector<int> children;
	Node(char x) :value(x) {

	}
};
void postorder(vector<Node> const &tree,vector<char> &ans, int root) {
	for (int i = 0; i < tree[root].children.size();i++ ) {
		postorder(tree, ans,tree[root].children[i]);
	}
	ans.push_back(tree[root].value);
}

int main() {
	int N;
	cin >> N;
	string line;
	getchar();
	vector<vector<Node> > trees;
	for (int t = 0; t < N; t++) {
		getline(cin, line);
		vector<char> values;
		vector<int> numbers;
		for (int i=0;i<line.size();i++) {
		    char c = line[i];
			if (isalpha(c)) {
				values.push_back(c);
			}
			else if (isalnum(c)) {
				numbers.push_back(c - '0');
			}
		}


		vector<Node> tree;
		int cur = 0;
		int childnum = numbers[0];
		int curnum = 0;
		tree.push_back(Node(values[0]));
		for (int i = 1; i < numbers.size(); i++) {
			if (curnum < childnum) {
				tree[cur].children.push_back(i);
				curnum++;
				tree.push_back(Node(values[i]));
			}
			else {
				cur++;
				childnum = numbers[cur];
				curnum = 0;
				i--;
			}
		}
		trees.push_back(tree);
	}
	vector<char> ans;
	for (int i=0;i<trees.size();i++)	     {
   vector<Node> tree = trees[i];
		postorder(tree,ans, 0);
	}
	bool first = true;
	for (int i=0;i<ans.size();i++) {
	    char a = ans[i];
		if (first){
			cout << a;
			first = false;
	}		
		else
			cout << " " << a;
	}
	cout << endl;

	return 0;
}