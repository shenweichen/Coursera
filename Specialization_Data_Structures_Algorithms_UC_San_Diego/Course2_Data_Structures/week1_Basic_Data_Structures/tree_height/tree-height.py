# python3

import sys, threading
sys.setrecursionlimit(10**7) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size

class TreeHeight(object):
    def __init__(self):
        self._n = 0
        self._children = []#using children representation
        self._parent =[]
        self._root = 0
    def read(self):
        self._n = int(sys.stdin.readline())
        for _ in range(0, self._n):
            self._children.append([])
        self._parent = [int(num) for num in sys.stdin.readline().split()]
        #list(map(int, sys.stdin.readline().split()))
        for index,parent in enumerate(self._parent):
            if(parent == -1):
                self._root = index
            else:
                self._children[parent].append(index)
        
    def compute(self,root):
        maxHeight = 0
        for index in range(0,len(self._children[root])):
            maxHeight = max(maxHeight,self.compute(self._children[root][index]))
        return maxHeight + 1

    def compute_height(self):
        # Replace this code with a faster implementation
        """
        maxHeight = 0
        for vertex in range(self.n):
                height = 0
                i = vertex
                while i != -1:
                        height += 1
                        i = self.parent[i]
                maxHeight = max(maxHeight, height);
        return maxHeight;
        """
        return self.compute(self._root)
            

def main():
    tree = TreeHeight()
    tree.read()
    print(tree.compute_height())

threading.Thread(target=main).start()
