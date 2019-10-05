# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:06:35 2019

@author: nikhil
"""
import random
import math
import copy

def getMinCuts(edges):
    uniqEdges = set(map(tuple,edges))
    if len(uniqEdges) <= 2:
        return len(edges)
    rei = int(random.uniform(0,len(edges)-1))
    re = edges[rei]
    v1=re[0]
    v2=re[1]
    for k in range(0,len(edges)):
        if v2 in edges[k]:
            if edges[k][0] == v2:
                edges[k][0] = v1
            else:
                edges[k][1] = v1
    edges2 = copy.deepcopy(edges)
    edges = list(filter(lambda i: i != re and i != [v1,v1], edges2))
    return getMinCuts(edges)
    
    
    

text_file = open("test.txt", "r")
arr = text_file.readlines()
g={}
edges=[]
for i in range(0,len(arr)):
    t=arr[i].split()
    v1=t[0]
    v2s=t[1:]
    for j in range(0,len(v2s)):
        edges.append([v1, v2s[j]])
#print(getMinCuts(edges))
minCuts=[]
edges1 = copy.deepcopy(edges)
p=len(edges)
for c in range(0,100):
    minCuts.append(getMinCuts(edges))
    edges = copy.deepcopy(edges1)
print(min(minCuts))