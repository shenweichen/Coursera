# python3

import sys, threading
from collections import deque
sys.setrecursionlimit(10**6) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size

class TreeOrders:
  def read(self):
    self.n = int(sys.stdin.readline())
    self.key = [0 for i in range(self.n)]
    self.left = [0 for i in range(self.n)]
    self.right = [0 for i in range(self.n)]
    self.isRoot = [0 for i in range(self.n)]
    for i in range(self.n):
      self.key[i],self.left[i],self.right[i] = map(int,sys.stdin.readline().split())

      #[a, b, c] = map(int, sys.stdin.readline().split())
      if(self.left[i]!=-1):
        self.isRoot[i]=1
      if(self.right[i]!=-1):
        self.isRoot[i]=1
      self.isRoot.sort()
      self.root =self.isRoot[0]
  def iter_inorder(self,root,r):
    st = []
    x = root
    while(True):
      while(x!=-1):
        st.append(x)
        x=self.left[x]
      if(len(st)==0):
        break
      x = st[-1]
      r.append(self.key[x])
      st.pop()
      x=self.right[x]
  
  def recur_inorder(self,root,r):
    if(self.left[root] != -1):
      self.recur_inorder(self.left[root],r)
    r.append(self.key[root])
    if(self.right[root]!=-1):
      self.recur_inorder(self.right[root],r)
      
  def iter_preorder(self,root,r):
    st = []
    x = root
    while(True):
      while(x!=-1):
        r.append(self.key[x])
        if(self.right[x]!=-1):
          st.append(self.right[x])
        x = self.left[x]
      if(len(st)==0):
        break
      x = st[-1]
      st.pop()

  def recur_preorder(self,root,r):
    r.append(self.key[root])
    if(self.left[root] != -1):
      self.recur_preorder(self.left[root],r)
    if(self.right[root]!=-1):
      self.recur_preorder(self.right[root],r)
  def iter_postorder(self,root,r):
    st = []
    x = root
    if(x!=-1):
      st.append(x)
    while(len(st)!=0):
      if(self.left[st[-1]]!=x and self.right[st[-1]]!=x):
        x = st[-1]
        while( x!=-1):
          if(self.left[x]!=-1):
            if(self.right[x]!=-1):
              st.append(self.right[x])
            st.append(self.left[x])
          else:
            st.append(self.right[x])
          x = st[-1]
        st.pop()
      x = st[-1]
      r.append(self.key[x])
      st.pop()
  
  def recur_postorder(self,root,r):
    if(self.left[root] != -1):
      self.recur_postorder(self.left[root],r)
    if(self.right[root] != -1):
      self.recur_postorder(self.right[root],r)
    r.append(self.key[root])
  def inOrder(self):
    self.result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    self.iter_inorder(self.root,self.result)
    #self.recur_inorder(self.root,self.result)
    return self.result

  def preOrder(self):
    self.result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    self.iter_preorder(self.root,self.result)   
    #self.recur_preorder(self.root,self.result)         
    return self.result

  def postOrder(self):
    self.result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    self.iter_postorder(self.root,self.result)     
    #self.recur_postorder(self.root,self.result)       
    return self.result

def main():
	tree = TreeOrders()
	tree.read()
	print(" ".join(str(x) for x in tree.inOrder()))
	print(" ".join(str(x) for x in tree.preOrder()))
	print(" ".join(str(x) for x in tree.postOrder()))

threading.Thread(target=main).start()
