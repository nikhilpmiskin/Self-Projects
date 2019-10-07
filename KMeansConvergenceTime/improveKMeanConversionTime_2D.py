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

k = 15

colr = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                     for i in range(k)]

n=data.shape[0]

time_taken = []

def getClosestCenter(x, centers):
    dist = []
    for i in centers:
        dist.append(np.linalg.norm(x-i))
    minC = min(dist)
    return dist.index(minC)

#Initial Plot
plt.scatter(X[:,0],X[:,1])

for KClusters in range(1,101):

    #Random normal kmeans
    
    start_time1 = time.time()
    
    kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(X)
    
    time_taken0 = (time.time() - start_time1)
    
    if KClusters == k:
        cs=[]
        for i in kmeans.labels_:
            cs.append(colr[getClosestCenter(kmeans.cluster_centers_[i], kmeans.cluster_centers_)])
        
        plt.title("Random normal method")
        plt.scatter(X[:,0],X[:,1], c=cs)
        plt.show()
    
    #Initial value normal kmeans
    
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
    kmeans1 = cluster.KMeans(n_clusters=KClusters, init = initialGuess, tol=pow(10,-4)).fit(X)
    
    time_taken1 = (time.time() - start_time2)
    
    if KClusters == k:
        cs=[]
        for i in kmeans1.labels_:
            cs.append(colr[getClosestCenter(kmeans1.cluster_centers_[i], kmeans.cluster_centers_)])
        
        plt.title("Initial Guess normal method")
        plt.scatter(X[:,0],X[:,1], c=cs)
        plt.show()
    
    #Random Batch Kmeans
    
    start_time3 = time.time()
    
    kmeans2 = cluster.MiniBatchKMeans(n_clusters=KClusters, batch_size = 10, random_state=0).fit(X)
    
    time_taken2 = (time.time() - start_time3)
    
    if KClusters == k:
        cs=[]
        for i in kmeans2.labels_:
            cs.append(colr[getClosestCenter(kmeans2.cluster_centers_[i], kmeans.cluster_centers_)])
        
        plt.title("Random Mini Batch method")
        plt.scatter(X[:,0],X[:,1], c=cs)
        plt.show()
    
    #Initial value Batch Kmeans
    
    start_time4 = time.time()
    
    kmeans3 = cluster.MiniBatchKMeans(n_clusters=KClusters, batch_size = 10, init=initialGuess).fit(X)
    
    time_taken3 = (time.time() - start_time4)
    
    if KClusters == k:
        cs=[]
        for i in kmeans3.labels_:
            cs.append(colr[getClosestCenter(kmeans3.cluster_centers_[i], kmeans.cluster_centers_)])
        
        plt.title("Initial Guess Mini Batch method")
        plt.scatter(X[:,0],X[:,1], c=cs)
        plt.show()
    
    #K++ Batch Kmeans
    
    start_time5 = time.time()
    
    kmeans4 = cluster.MiniBatchKMeans(n_clusters=KClusters, batch_size = 10, init='k-means++').fit(X)
    
    time_taken4 = time.time() - start_time5
    
    if KClusters == k:
        cs=[]
        for i in kmeans4.labels_:
            cs.append(colr[getClosestCenter(kmeans4.cluster_centers_[i], kmeans.cluster_centers_)])
        
        plt.title("K++ initial guess Mini Batch method")
        plt.scatter(X[:,0],X[:,1], c=cs)
        plt.show()
    
    #K++ normal kmeans
    
    start_time6 = time.time()
    
    kmeans5 = cluster.KMeans(n_clusters=KClusters, init='k-means++').fit(X)
    
    time_taken5 = (time.time() - start_time6)
    
    if KClusters == k:
        cs=[]
        for i in kmeans5.labels_:
            cs.append(colr[getClosestCenter(kmeans5.cluster_centers_[i], kmeans.cluster_centers_)])
        
        plt.title("K++ initial guess normal method")
        plt.scatter(X[:,0],X[:,1], c=cs)
        plt.show()
    time_taken.append([time_taken0, time_taken1, time_taken2, time_taken3, time_taken4, time_taken5])

labels=["Random Initialization", "Initial Guess normal method", "Random Mini Batch method", "Initial Guess Mini Batch method",
        "K++ initial guess Mini Batch method", "K++ initial guess normal method"]

for i in range(0,6):
    plt.plot(range(1,101), np.array(time_taken)[:,i], label=labels[i])
plt.gcf().set_size_inches(10, 8)
plt.xlabel('K Values')
plt.ylabel('Time taken (in s)')
plt.legend()
plt.show()


comp = []
comp.append(["Random Normal Method", time_taken[k-1][0]])
comp.append(["Initial Guess normal method", time_taken[k-1][1]])
comp.append(["Random Mini Batch method", time_taken[k-1][2]])
comp.append(["Initial Guess Mini Batch method", time_taken[k-1][3]])
comp.append(["K++ initial guess Mini Batch method", time_taken[k-1][4]])
comp.append(["K++ initial guess normal method", time_taken[k-1][5]])