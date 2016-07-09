#include <iostream>
#include <vector>
using namespace std;

int main()
{
    //输入部分
    int n = 0;
    cin >> n;
    vector<float> rate(n);
    vector<int> id(n);
    for (int i = 0; i < n; i++)
    {
        int first = 0, last = 0;
        cin >> id[i] >> first >> last;
        rate[id[i]-1] = (float)last/first;
    }

    //繁殖率升序排序
    for (int i = 0; i < n-1; i++)
    {
        for (int j = i+1; j < n; j++)
        {
            if (rate[id[i]-1] > rate[id[j]-1])
            {
                int temp = id[i];
                id[i] = id[j];
                id[j] = temp;
            }
        }
    }

    int maxDifference = 0;
    int flagSplit = 0;
    for (int i = 0; i < n-1; i++)
    {
        if (maxDifference < rate[id[i+1]-1]-rate[id[i]-1])
        {
            maxDifference = rate[id[i+1]-1]-rate[id[i]-1];
            flagSplit = i;
        }
    }

    //输出
    cout << n-flagSplit-1 << endl;
    for (int i = flagSplit+1; i < n; i++)
    {
        cout << id[i] << endl;
    }
    cout << flagSplit+1 << endl;
    for (int i = 0; i < flagSplit+1; i++)
    {
        cout << id[i] << endl;
    }
    return 0;
}


/*#include <iostream>
#include <algorithm>
using namespace std;
struct D
{
	int id;
	int before;
	int after;
	int rate;
}d[100];

bool cmp(D a,D b){
	return a.rate<b.rate;
}

int main() {
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		cin>>d[i].id>>d[i].before>>d[i].after;
		d[i].rate=d[i].after-d[i].before;
	}
	sort(d,d+n,cmp);

	int count ;
	for(int i=1;i<n-1;i++){
		if(d[i+1].rate-d[i].rate>d[i].rate-d[0].rate){
			count = i+1;
			break;
		}
	}


	cout<<n-count<<endl;
	for(int i=count;i<n;i++)
		cout<<d[i].id<<endl;
	cout<<count<<endl;
	for(int i=0;i<count;i++)
		cout<<d[i].id<<endl;


}*/