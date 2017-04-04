from __future__ import print_function
from __future__ import division
from tqdm import tqdm

class Apriori:
    def __init__(self):
        self.dataSet = None

    def loadData(self,filepath='categories.txt'):
        with open(filepath, 'r') as f:
            self.dataSet = [line.strip().split(';') for line in f.readlines()]



    def createC1(self ,):
        """
        创建候选单项集合
        :param dataset: 
        :return: 
        """
        if self.dataSet == None:
            print("have not load data")
            return None

        C1 = []

        for transaction in self.dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()
        return  [frozenset(x) for x in C1]




    def scanD(self,D,Ck,minSupport):
        """
        创建频繁项集
        :param D: 
        :param Ck: 
        :param minSupport: 
        :return: 
        """

        ssCnt = {}

        for tid in tqdm(D):
            for can in Ck:
                if can.issubset(tid):
                    if not can in ssCnt.keys():
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1

        numItems = len(D)
        retList = []
        superData = {}
        for key in ssCnt:
            support = ssCnt[key]/numItems
            if support >= minSupport:
                retList.insert(0,key)
            superData[key] = ssCnt[key]
        return retList,superData


    def aprioriGen(self,Lk,k):
        """
        根据频繁项集建候选项集Ck
        :param Lk: 频繁项集
        :param k: 
        :return: 
        """
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk):
            for j in range(i+1,lenLk):
                L1 = list(Lk[i])[:k-2];L2 = list(Lk[j])[:k-2]
                L1.sort();L2.sort()
                if L1==L2:
                    retList.append(Lk[i]|Lk[j])
        return retList

    def apriori(self,minSupport = 0.01,K=1e9):
        C1 = self.createC1()
        D = list(map(set,self.dataSet))
        L1,supportData = self.scanD(D,C1,minSupport)
        L = [L1]
        k=2
        while(len(L[k-2])>0 and k<=K):
            Ck = self.aprioriGen(L[k-2],k)
            Lk,supK = self.scanD(D,Ck,minSupport)
            supportData.update(supK)
            L.append(Lk)
            k +=1

        self.L = L
        self.supportData = supportData




    def writeout(self,filename='patterns_all.txt',K=1e9,):
        """
        :param filename filepath
        :param K frequent k itemset
        """
        with open(filename,'w') as f:
            for  k,Lk in enumerate(self.L ,start=1):
                if k <=K:
                    for item in Lk:
                        f.write(str(self.supportData[item])+':'+';'.join(list(item))+'\n')


a = Apriori()
a.loadData()
a.apriori(0.01)
a.writeout()