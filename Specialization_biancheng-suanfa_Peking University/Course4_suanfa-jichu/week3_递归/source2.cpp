#include <iostream>
#include <string>
#include <stack>
#include <algorithm>
#include <cstring>
#include <queue>
using namespace std;
stack <int> st;
int total, current, sq = 0;
struct FILES
{
	string name;
	int childid[20];
	int childnum = 0;
	bool isDir = true;
	bool match = false;
}file[1000];
inline void PrintFilename(int l, const string &s) {
	for (int i = 0; i < l; i++)
		cout << "|     ";
	cout << s << endl;
}
class cmp {
public:
	bool operator()(string a, string b) {
		return a > b;
	}
};

void PrintFile(int id, int l) {
	struct FILES  curfile = file[id];
	PrintFilename(l, curfile.name);
	priority_queue<string, vector<string>, cmp> q;
	for (int i = curfile.childnum - 1; i >= 0; i--) {
		int child_id = curfile.childid[i];
		if (file[child_id].isDir)
			PrintFile(child_id, l + 1);
		else
			q.push(file[child_id].name);
	}
	while (!q.empty())
	{
		PrintFilename(l, q.top());
		q.pop();
	};
	return;

}
void init() {
	total = current = 0;
	for (int i = 0; i < 1000; i++) {
		file[i].childnum = 0;
		file[i].isDir = true;
		file[i].match = false;
	}
	while (!st.empty()) {
		st.pop();
	}

};
int main() {
	string s;
	init();
	while (cin >> s) {
		if (s[0] == '#')
			break;
		if (s[0] == '*') {
			sq++;
			file[current].name = "ROOT";
			while (!st.empty()) {
				file[current].childid[file[current].childnum++] = st.top();
				st.pop();
			}
			cout << "DATA SET " << sq << ":" << endl;
			PrintFile(current, 0);
			cout << endl;
			init();
		}
		else {

			if (s[0] == ']') {
				int tempchild[20];
				int tempnum = 0;
				while (!st.empty())
				{
					struct FILES& topfile = file[st.top()];
					if (topfile.isDir == false || topfile.match == true) {
						tempchild[tempnum++] = st.top();
						st.pop();
					}
					else {
						topfile.match = true;
						for (int i = 0; i < tempnum; i++)
							topfile.childid[topfile.childnum++] = tempchild[i];
						break;
					}
				}

			}
			else {
				file[current].name = s;
				total++;
				if (s[0] == 'f')
					file[current].isDir = false;
				st.push(current);
				current = total;
			}
		}

	}


}