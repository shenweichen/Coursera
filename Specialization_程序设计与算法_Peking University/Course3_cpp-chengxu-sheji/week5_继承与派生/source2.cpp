#include <iostream>
#include <cstring>
#include <iomanip>
using namespace std;
int squence[2][5]{ {2,3,4,1,0},{3,0,1,2,4} };
char headname[2][5] = { "red","blue" };
char warrior_name[5][7] = { "dragon","ninja","iceman","lion","wolf" };
char weapon_name[3][6]={"sword","bomb","arrow"};

class warrior// dragon0 、ninja1、iceman2、lion3、wolf 4
{
	int id;
	int strength;
public:
	warrior(int id,int strength):id(id),strength(strength){};
	
	
};
class dragon:public warrior
{
	int weapon_id;
	double morale;
public:
	dragon(int id,int strength,int remain_strength):warrior(id,strength){
		weapon_id = id%3;
		morale = (double)remain_strength/strength;
		cout<<"It has a "<<weapon_name[weapon_id]<<",and it's morale is "<<fixed<<setprecision(2)<<morale<<endl;
	};
	
	
};
class ninja:public warrior
{
	int weapon_id_1;
	int weapon_id_2;
public:
	ninja(int id,int strength):warrior(id,strength){
		weapon_id_1=id%3;
		weapon_id_2=(id+1)%3;
		cout<<"It has a "<<weapon_name[weapon_id_1]<<" and a "<<weapon_name[weapon_id_2]<<endl;
	};
	
	
};
class iceman:public warrior
{
	int weapon_id;
public:
	iceman(int id,int strength):warrior(id,strength){
		weapon_id = id%3;
		cout<<"It has a "<<weapon_name[weapon_id]<<endl;
	};
	
	
};
class lion:public warrior
{
	int loyalty;
public:
	lion(int id,int strength,int remain_strength):warrior(id,strength){
		loyalty = remain_strength;
		cout<<"It's loyalty is "<<loyalty<<endl;
	};
	
	
};
class wolf:public warrior
{
public:
	wolf(int id,int strength):warrior(id,strength){

	};
	
	
};

class headquarter
{
	int num[5];// dragon0 、ninja1、iceman2、lion3、wolf 4
	int total_strength;
	int strength[5];
	int id;
	int current;
	int warrior_id;
	int cant[5];
	int sumcant;
public:
	headquarter(int ID, int M, int str[]) {
		id = ID;
		warrior_id = 0;
		current = 0;
		memset(num, 0, 5 * sizeof(int));
		memset(cant, 0, 5 * sizeof(int));
		sumcant = 0;
		total_strength = M;
		for (int i = 0; i < 5; i++)
			strength[i] = str[i];
	};
	bool canmake() {
		if(sumcant==5)
			return false;
		else
			return true;
	}
	void Printwarrior(int need_id,int id){
		switch(need_id){// dragon0 、ninja1、iceman2、lion3、wolf 4
		case 0: {dragon dg(id, strength[need_id], total_strength); break; }
		case 1: {ninja nj(id, strength[need_id]); break; }
		case 2: {iceman im(id, strength[need_id]); break; }
		case 3: {lion ln(id, strength[need_id], total_strength); break; }
		case 4: {wolf wf(id, strength[need_id]); break; }
		}
	}
	bool makewarriors() {
		int need_id = squence[id][current];
		if (total_strength >= strength[need_id]) {
			num[need_id]++;
			total_strength -= strength[need_id];
			warrior_id++;
			cout << headname[id] << " " << warrior_name[need_id] << " " << warrior_id << " born with strength " << strength[need_id] << "," << num[need_id] << " " << warrior_name[need_id] << " in " << headname[id] << " headquarter" << endl;
			Printwarrior(need_id,warrior_id);
			current = (current+1) % 5;
			return false;
		}
		else if (cant[need_id] == 0) {
			cant[need_id] = 1;
			sumcant++;
			if (sumcant == 5) {
				cout << headname[id] << " headquarter stops making warriors" << endl;
				return false;
			}
		}
		current = (current+1) % 5;
		return true;

	};

};



int main() {
	int casenum,m; int n[5];
	cin>>casenum;
	for(int caseid=1;caseid<=casenum;caseid++){
				cin >> m;
				for (int i = 0; i < 5; i++)
					cin >> n[i];
				cout<<"Case:"<<caseid<<endl;
				headquarter red(0, m, n);
				headquarter blue(1, m, n);
				int time = 0;

				while (red.canmake()||blue.canmake()) {
					if (red.canmake()) {
						cout << setw(3) << setfill('0') << time << " ";
						while (red.makewarriors());
							}
					if (blue.canmake()) {
						cout << setw(3) << setfill('0') << time << " ";
						while (blue.makewarriors()); 			
					}
					time++;
				};
			}
}