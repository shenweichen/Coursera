# python3

import sys

n, m = map(int, sys.stdin.readline().split())
lines = list(map(int, sys.stdin.readline().split()))
rank = [1] * n
parent = list(range(0, n))
ans = max(lines)

def getParent(table):
    # find parent and compress path
    if(parent[table] != table):
        parent[table] = getParent(parent[table])
    return parent[table]

def merge(destination, source):
    realDestination, realSource = getParent(destination), getParent(source)
    # merge two components
    # use union by rank heuristic 
    # update ans with the new maximum table size
    if(realDestination != realSource):
        if(rank[realDestination]<rank[realSource]):
            parent[realDestination] = realSource
            lines[realSource] += lines[realDestination]
            lines[realDestination] = 0
        else:
            parent[realSource] = realDestination
            lines[realDestination] += lines[realSource]
            lines[realSource] = 0
            if(rank[realDestination] == rank[realSource]):
                rank[realDestination]+=1
        global ans
        ans = max(ans,lines[realDestination]+lines[realSource])

for i in range(m):
    destination, source = map(int, sys.stdin.readline().split())
    merge(destination - 1, source - 1)
    print(ans)
    
