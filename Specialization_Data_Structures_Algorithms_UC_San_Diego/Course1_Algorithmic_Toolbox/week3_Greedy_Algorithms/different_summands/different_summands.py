# Uses python3
import sys

def optimal_summands(n):
    summands = []
    #write your code here
    if n == 1:
        return [1]
    for l in range(1,n):
        if n <= 2*l:
            summands.append(n)
            break
        else:
            summands.append(l)
            n-=l

    return summands

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    summands = optimal_summands(n)
    print(len(summands))
    for x in summands:
        print(x, end=' ')
