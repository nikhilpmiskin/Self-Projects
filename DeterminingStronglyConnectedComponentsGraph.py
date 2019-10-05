# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:03:31 2019

@author: nikhil
"""
finishTime=0
sizes=[]
size=1

class Gnode:
    def __init__(self, value, explored, toEdges):
        self.value = value
        self.explored = explored
        self.toEdges = toEdges
        self.newVal=0
        self.inEdges=[]

def dfs(graphNodes, n):
    global finishTime
    global size
    n.explored = True
    for v in n.toEdges:
        if graphNodes[v-1].explored == False:
            size+=1
            dfs(graphNodes,graphNodes[v-1])
    finishTime+=1
    n.newVal = finishTime

tf = open("graphData.txt", "r")
graphNodes={}
v=0
vals=[]
for line in tf.readlines():
    w = line.split()
    v=int(w[0])
    v1=int(w[1])
#    prevNodes = list(filter(lambda x: x[1].value == v, graphNodes.items()))
    if v not in vals:
        gn = Gnode(v, False, [])
        graphNodes[v-1]=gn
#    prevNodesV = list(filter(lambda x: x[1].value == v1, graphNodes.items()))
    if v1 not in vals:
        gn = Gnode(v1, False, [])
        graphNodes[v1-1]=gn
    graphNodes[v-1].toEdges.append(int(w[1]))
    vals.append(v)

#for i in graphNodes.values():
#    for j in i.toEdges:
#        graphNodes[j-1].inEdges.append(i.value)
#            
##reverse edges
#for i in graphNodes.values():
#    temp = i.inEdges
#    i.inEdges=i.toEdges
#    i.toEdges = temp
#
#
#for i in range(len(graphNodes)-1,-1,-1):
#    if graphNodes[i].explored == False:
#        dfs(graphNodes, graphNodes[i])
#
#sortG = sorted(graphNodes, key=lambda x: graphNodes[x].newVal, reverse=True)
#
#
##reverse edges
#for i in graphNodes.values():
#    temp = i.inEdges
#    i.inEdges=i.toEdges
#    i.toEdges = temp
#    i.explored = False
#
#
#finishTime=0
#sizes=[]
#size=1
#
#for i in sortG:
#    if graphNodes[i].explored == False:
#        dfs(graphNodes, graphNodes[i])
#        sizes.append(size)
#        size=1