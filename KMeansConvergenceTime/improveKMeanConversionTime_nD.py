import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd
import time

data = pd.read_csv('dim032.txt', sep=",", header=None)

l = list(data.values)

X = np.array(l)

KClusters = 2

n=data.shape[0]

start_time1 = time.time()

kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(X)

time_taken1 = time.time() - start_time1

minC = np.amin(X,axis=0)
maxC = np.amax(X,axis=0)

start_time2 = time.time()

initialGuess=[]
for j in range(0,len(l[0])):
    ig = []
    i=minC[j]
    slope = (maxC[j] - minC[j]) / KClusters
    while i < (maxC[j]-pow(10,-6)):
        ig.append(i + (slope)/2)
        i += slope
    initialGuess.append(ig)

initialGuess = np.array(initialGuess).transpose()
kmeansI = cluster.KMeans(n_clusters=KClusters, init = initialGuess, tol=pow(10,-10)).fit(X)

time_taken2 = time.time() - start_time2

percentFaster = (time_taken1 - time_taken2) *100 / (time_taken1)
print("\n\nSorted is faster than random by " + str(round(percentFaster,2)) + "%")


