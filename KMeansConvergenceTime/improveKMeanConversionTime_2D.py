import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

data = pd.read_csv('s1.txt', sep=",", header=None)
data.columns = ["x", "y"]
l = list(data.values)
X = np.array(l)

KClusters = 4

n=data.shape[0]

#Initial Plot
plt.scatter(X[:,0],X[:,1])

start_time1 = time.time()

kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(X)

time_taken1 = time.time() - start_time1

plt.title("Random method")
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()

minx = min(X[:,0])
miny = min(X[:,1])

maxx = max(X[:,0])
maxy = max(X[:,1])

start_time2 = time.time()

initialGuessX = []
initialGuessY = []
slopex = (maxx - minx) / KClusters
slopey = (maxy - miny) / KClusters
ix = minx
iy = miny
while ix < (maxx-pow(10,-6)):
    initialGuessX.append(ix + (slopex)/2)
    ix += slopex
while iy < (maxy-pow(10,-6)):
    initialGuessY.append(iy + (slopey)/2)
    iy += slopey
initialGuess = np.array([initialGuessX, initialGuessY]).transpose()
kmeansI = cluster.KMeans(n_clusters=KClusters, init = initialGuess, tol=pow(10,-4)).fit(X)

time_taken2 = time.time() - start_time2

percentFaster = (time_taken1 - time_taken2) *100 / (time_taken1)
print("\n\nSorted is faster than random by " + str(round(percentFaster,2)) + "%")

plt.title("Initial Guess method")
plt.scatter(X[:,0],X[:,1], c=kmeansI.labels_, cmap='rainbow')
plt.show()

