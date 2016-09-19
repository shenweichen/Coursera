# Uses python3
#from __future__ import print_function
def calc_fib(num):
    if (num <= 1):
        return num
    a = 0
    b = 1
    temp = 1
    for i in range(2, num+1):
        temp = a + b
        a = b
        b = temp
    return temp


n = int(input())
print(calc_fib(n))
