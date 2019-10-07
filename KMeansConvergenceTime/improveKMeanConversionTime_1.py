import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import time
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
import random

style.use('ggplot')

DistanceFromChicago = pandas.read_csv('DistanceFromChicago.csv',
                      delimiter=',', index_col='CityState')

nCity = DistanceFromChicago.shape[0]

trainData = numpy.reshape(numpy.asarray(DistanceFromChicago['DrivingMilesFromChicago']), (nCity, 1))
N=15
colr = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                     for i in range(N)]
kmN = 0

def getClosestCenter(x, centers):
    dist = []
    for i in centers:
        dist.append(abs(x-i))
    minC = min(dist)
    return dist.index(minC)

timeRnd = []
for kc in range(1,60):
    start_time1 = time.time()
    KClusters = kc
    
    #n_init is 1 as we are checking time required to reach convergence once
    kmeans1 = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(trainData)
    
    time_taken1 = time.time() - start_time1
    timeRnd.append(time_taken1)
    
    #Plotting on map for k=N
    if KClusters == N:
        kmN = kmeans1
        DistanceFromChicago['KMeanCluster1'] = kmeans1.labels_
        
        map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
        
        # load the shapefile, use the name 'states'
        map.readshapefile('st99_d00', name='states', drawbounds=True)
        # Get the location of each city and plot it
        geolocator = Nominatim()
        loc1=[]
        rgb=[]
        for index, row in DistanceFromChicago.iterrows():
            city = row['City']
            loc = geolocator.geocode(city)
            x, y = map(loc.longitude, loc.latitude)
            loc1.append([x,y])
            rgb.append(colr[getClosestCenter(kmeans1.cluster_centers_[row['KMeanCluster1']], kmeans1.cluster_centers_)])
        X=[i[0] for i in loc1]
        Y=[i[1] for i in loc1]
        map.scatter(X, Y, c=rgb, s=50)
        loc = geolocator.geocode("Chicago")
        x, y = map(loc.longitude, loc.latitude)
        map.scatter(x,y,s=100)
        plt.title("Plot with Random initial values")
        plt.show()
        for k in range(KClusters):
            print("Cluster ", k)
            print(DistanceFromChicago[kmeans1.labels_ == k])
    
print("Time taken for random intial guess " + str(time_taken1))
print("Kmeans with random initial values, the means are")
print(numpy.sort(kmeans1.cluster_centers_, axis=None))

tDataMin = int(min(trainData))
tDataMax = int(max(trainData))

timeInit = []
for kc in range(1,60):
    KClusters=kc
    start_time = time.time()
    initialGuess = []
    slope = (tDataMax - tDataMin) / KClusters
    i = tDataMin
    while i < (tDataMax-pow(10,-6)):
        initialGuess.append(i + (slope)/2)
        i += slope
        
    initGuessArr = numpy.reshape(numpy.array(initialGuess), (KClusters, 1))
    
    kmeans = cluster.KMeans(n_clusters=KClusters, init = initGuessArr).fit(trainData)
    
    time_taken2 = time.time() - start_time
    timeInit.append(time_taken2)
    
    #Plotting on map for k=N
    if KClusters == N:
        DistanceFromChicago['KMeanCluster'] = kmeans.labels_
    
        map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
        
        # load the shapefile, use the name 'states'
        map.readshapefile('st99_d00', name='states', drawbounds=True)
        # Get the location of each city and plot it
        geolocator = Nominatim()
        loc1=[]
        rgb=[]
        for index, row in DistanceFromChicago.iterrows():
            city = row['City']
            loc = geolocator.geocode(city)
            x, y = map(loc.longitude, loc.latitude)
            loc1.append([x,y])
            rgb.append(colr[getClosestCenter(kmeans.cluster_centers_[row['KMeanCluster']], kmN.cluster_centers_)])
        X=[i[0] for i in loc1]
        Y=[i[1] for i in loc1]
        map.scatter(X, Y, c=rgb, s=50)
        loc = geolocator.geocode("Chicago")
        x, y = map(loc.longitude, loc.latitude)
        map.scatter(x,y,s=100)
        plt.title("Plot with initial value defined")
        plt.show()
        
        for k in range(KClusters):
            print("Cluster ", k)
            print(DistanceFromChicago[kmeans.labels_ == k])
    
    
print("Time taken for sorted intial guess " + str(time_taken2))
print("Kmeans with mid points of K equal splits as initial values, the means are")
print(kmeans.cluster_centers_)

percentFaster = (time_taken1 - time_taken2) *100 / (time_taken1)
print("Sorted is faster than random by " + str(round(percentFaster,2)) + "%")

plt.plot(range(1,60), timeRnd, label="Random Initialization")
plt.plot(range(1,60), timeInit, label="Midpoint Initialization")
plt.xlabel('K Values')
plt.ylabel('Time taken (in s)')
plt.legend()
plt.show()

