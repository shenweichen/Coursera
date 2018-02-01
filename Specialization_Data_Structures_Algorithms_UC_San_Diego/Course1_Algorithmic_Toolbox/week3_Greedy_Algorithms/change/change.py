# Uses python3
import sys

def get_change(m):
    #write your code here
    coins = 0
    for coin in [10,5,1]:
        coins += m//coin
        m%=coin
    m = coins
    return m

if __name__ == '__main__':
    m = int(sys.stdin.read())
    print(get_change(m))
