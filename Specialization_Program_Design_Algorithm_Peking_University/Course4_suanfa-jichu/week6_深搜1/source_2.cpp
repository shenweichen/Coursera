#include <cstdio>
#include <cstring>

using namespace std;

const int MAX_N = 27;
//字典序最小的行走方向
const int dx[8] = {-1, 1, -2, 2, -2, 2, -1, 1}; 
const int dy[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
bool visited[MAX_N][MAX_N];
struct Step{
    char x, y;
} path[MAX_N];
bool success;           //是否成功遍历的标记
int cases, p, q;

void DFS(int x, int y, int num);

int main()
{
    scanf("%d", &cases);
    for (int c = 1; c <= cases; c++)
    {
        success = false;
        scanf("%d%d", &p, &q);
        memset(visited, false, sizeof(visited));
        visited[1][1] = true;    //起点
        DFS(1, 1, 1);              
        printf("Scenario #%d:\n", c);
        if (success)
        {
            for (int i = 1; i <= p * q; i++)
                printf("%c%c", path[i].y, path[i].x);
            printf("\n");
        }
        else
            printf("impossible\n");
        if (c != cases)
            printf("\n");      //注意该题的换行
    }
    return 0;
}

void DFS(int x, int y, int num)
{
    path[num].y = y + 'A' - 1;   //int 转为 char
    path[num].x = x + '0';
    if (num == p * q) 
    {
        success = true;
        return;
    }
    for (int i = 0; i < 8; i++)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (0 < nx && nx <= p && 0 < ny && ny <= q
            && !visited[nx][ny] && !success)
        {
            visited[nx][ny] = true;
            DFS(nx, ny, num+1);
            visited[nx][ny] = false;    //撤销该步
        }
    }
}