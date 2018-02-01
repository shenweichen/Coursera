#Uses python3

import sys
def cmp(x,y):
    return x+y > y+x
    
def cmp2(x,y):
    if len(x) == len(y):
        return x>y
    i = 0
    j = 0
    while(i<len(x) and j<len(y)):
        if x[i] >y[j]:
            return True
        elif x[i]<y[j]:
            return False
        i+=1
        j+=1
        
    if i == len(x):#x遍历结束
        minsize = min(len(y)-j,i)
        if x[:minsize] > y[j:j+minsize] :
            return True
        elif x[:minsize] < y[j:j+minsize] :
            return False
        else:
            return True
    else:
        minsize = min(len(x)-i,j)
        if x[i:i+minsize] > y[:minsize]:
            return True
        elif x[i:i+minsize] < y[:minsize]:
            return False
        else:
            return False

def largest_number(a):
    #write your code here
    
    res = ""
    while(len(a) >0):
        max_num = None
        for num in a:
            if max_num is None:
                max_num = num
            elif cmp(num,max_num):
                max_num = num
        res += max_num
        a.remove(max_num)
    #for x in a:
    #    res += x
    return res

if __name__ == '__main__':
    input = sys.stdin.read()
    data = input.split()
    a = data[1:]
    print(largest_number(a))
    
