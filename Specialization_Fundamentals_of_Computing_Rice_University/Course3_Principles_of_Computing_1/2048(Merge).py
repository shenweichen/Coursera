"""
Merge function for 2048 game.
"""

def merge(line):
    """
    Function that merges a single row or column in 2048.
    """
    # replace with your code
    ans =  [0]*len(line)
    pre = 0
    for cur in range(0,len(line)):
        if(line[cur] == 0):
            continue
        if( ans[pre] == 0):
            ans[pre] = line[cur]
        elif(line[cur] == ans[pre]):
            ans[pre] += line[cur]
            pre += 1
        else:
            pre += 1
            ans[pre] = line[cur]
        

    return ans