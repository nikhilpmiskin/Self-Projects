

tf = open("dijkstraData.txt", "r")
edges=[]
X=[]
sizes={}

for line in tf.readlines():
    pairs = line.split("\t")
    v = pairs[0]
    for i in range(1,len(pairs)-1):
        edge={}
        edge["v"] = int(v)
        edgLen = pairs[i].split(",")
        edge["w"] = int(edgLen[0])
        edge["l"] = int(edgLen[1])
        edges.append(edge)
firstV = edges[0]["v"]        
X.append(firstV)
sizes[firstV] = 0

while not not edges:
    smEdg = {}
    l=-1
    
    for edge in edges:
        if edge["v"] in X and edge["w"] not in X:
            if (sizes[edge["v"]] + edge["l"] <= l) or l==-1:
                smEdg = edge
                l=sizes[edge["v"]] + edge["l"]
    if l != -1:
        X.append(smEdg["w"])
        sizes[smEdg["w"]] = l
        edges.remove(smEdg)
    else:
        break

prnt=""
for i in [7,37,59,82,99,115,133,165,188,197]:
    prnt = prnt + str(sizes[i]) + ","
