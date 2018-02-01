# Uses python3
import sys

def get_optimal_value(capacity, weights, values):
    value = 0.
    # write your code here
    dic = [(i,values[i]/weights[i]) for i in range(len(weights))]
    dic.sort(key=lambda x:x[1],reverse=True)
    for item in dic:
        if weights[item[0]] <= capacity:
            value += values[item[0]]
            capacity -= weights[item[0]]
        else:
            value += (capacity*item[1])
            break


    return value


if __name__ == "__main__":
    data = list(map(int, sys.stdin.read().split()))
    n, capacity = data[0:2]
    values = data[2:(2 * n + 2):2]
    weights = data[3:(2 * n + 2):2]
    opt_value = get_optimal_value(capacity, weights, values)
    print("{:.10f}".format(opt_value))
