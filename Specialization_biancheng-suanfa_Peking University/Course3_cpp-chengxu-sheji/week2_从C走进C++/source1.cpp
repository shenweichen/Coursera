#include <iostream>
#include <cstdio>
using namespace std;
class Student
{
	char name[10];
	char id[10];
	int age;
	int score[4];
	int sum;
public:

	Student() {
		sum = 0;
	};
	void init() {
		char c;
		cin.getline(name, 10, ',');
		cin >> age;
		getchar();
		cin.getline(id, 10, ',');
		for (int i = 0; i < 4; i++) {
			if (i < 3)
				cin >> score[i] >> c;
			else
				cin >> score[i];
		}

	};
	void show() {
		char dot = ',';
		for (int i = 0; i<4; i++)
			sum += score[i];
		cout << name << dot << age << dot << id << dot << sum / 4 << endl;
	};
};

int main(){

	Student s;
	s.init();
	s.show();
}