/*#include <iostream>
#include <string>
#include <map>
#include <iterator>
#include <algorithm>
using namespace std;*/

template<class Key,class T,class Pred = greater<Key> >
class MyMultimap:public multimap<Key,T,Pred>
{
public:
    MyMultimap():multimap<Key,T,Pred>(){}
    void Set(const Key &key, const T &t){       
        for(typename MyMultimap<Key,T,Pred>::iterator it = this->begin();it!=this->end();++it){
            if(it->first==key)
                it->second=t;
        }
    }
        template<class Key_,class T_>
    friend ostream& operator<<(ostream &out, const pair< Key_, T_> &a) ;//友元函数不是类的成员函数，不能使用类模板的类型参数
};

template<class Key,class T>
 ostream& operator<<(ostream &out,const pair< Key,T> &a){
       out << "(" << a.first << "," << a.second << ")";
   return out;
   }

/*
struct Student
{
        string name;
        int score;
};
template <class T>
void Print(T first,T last) {
        for(;first!= last; ++ first)
                cout << * first << ",";
        cout << endl;
}
int main()
{

        Student s[] = { {"Tom",80},{"Jack",70},
                                        {"Jone",90},{"Tom",70},{"Alice",100} };
        MyMultimap<string,int> mp;
        for(int i = 0; i<5; ++ i)
                mp.insert(make_pair(s[i].name,s[i].score));

        Print(mp.begin(),mp.end()); //按姓名从大到小输出

        mp.Set("Tom",78); //把所有名为"Tom"的学生的成绩都设置为78
        Print(mp.begin(),mp.end());


        MyMultimap<int,string,less<int> > mp2;
        for(int i = 0; i<5; ++ i)
                mp2.insert(make_pair(s[i].score,s[i].name));

        Print(mp2.begin(),mp2.end()); //按成绩从小到大输出
        mp2.Set(70,"Error");          //把所有成绩为70的学生，名字都改为"Error"
        Print(mp2.begin(),mp2.end());
        cout << "******" << endl;

       mp.clear();
        
        string name;
        string cmd;
        int score;
        while(cin >> cmd ) {
                if( cmd == "A") {
                        cin >> name >> score;
                        if(mp.find(name) != mp.end() ) {
                                cout << "erroe" << endl;
                        }
                        mp.insert(make_pair(name,score));
                }
                else if(cmd == "Q") {
                        cin >> name;
                        MyMultimap<string,int>::iterator p = mp.find(name);
                        if( p!= mp.end()) {
                                cout << p->second << endl;
                        }
                        else {
                                cout << "Not Found" << endl;
                        }
                }
        }
        return 0;
}*/