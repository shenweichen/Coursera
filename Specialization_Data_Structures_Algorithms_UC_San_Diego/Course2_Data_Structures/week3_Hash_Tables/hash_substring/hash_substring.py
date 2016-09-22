# python3
import random
import sys
def read_input():
    return (input().rstrip(), input().rstrip())

def print_occurrences(output):
    print(' '.join(map(str, output)))
#TODO:
def hash_func(s,prime,multiplier):
    hash_ = 0
    for c in reversed(s):
        hash_ = (hash_ * multiplier + ord(c))%prime
    return hash_

def PrecomputeHashes(T,lenP,p,x):
    lenT = len(T)
    H = [0]*(lenT-lenP+1)
    S = T[lenT-lenP:]
    H[lenT-lenP] = hash_func(S,p,x)
    y =1
    for _ in range(1,lenP+1):
        y = (y*x)%p
    for i in range(lenT - lenP - 1,-1,-1):
        H[i] = ((x * H[i + 1] + ord(T[i]) - y * ord(T[i + lenP])) % p + p) % p
    return H

def AreEqual(T,P,start): 
    # for i in range(start,start+len(P)):  too slow
    #     if(T[i]!=P[i-start]):
    #         return False
    # return True
    return T[start:start+len(P)] == P

def RabinKarp(P,T):
    p = int(1e9) + 7 
    x = random.randint(1,p-1) 
    result = []
    pHash = hash_func(P,p,x)
    H = PrecomputeHashes(T,len(P),p,x)
    for i in range(0,len(T)-len(P)+1):
        if(pHash!=H[i]):
            continue
        if(AreEqual(T,P,i)):
            result.append(i)
    return result

def get_occurrences(pattern, text):
    return [
        i 
        for i in range(len(text) - len(pattern) + 1) 
        if text[i:i + len(pattern)] == pattern
    ]

if __name__ == '__main__':
   # print_occurrences(get_occurrences(*read_input()))
    print_occurrences(RabinKarp(*read_input()))

