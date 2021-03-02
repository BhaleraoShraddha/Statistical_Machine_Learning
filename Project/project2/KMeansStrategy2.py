import scipy.io
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


NumberOfClusters = 2
ObjectiveFuncList = []

#Extract Data from the matlab data file
Numpyfile = scipy.io.loadmat('AllSamples.mat')
SampleData = Numpyfile['AllSamples']
# print(SampleData.shape)

#Plot the Sample points before Clustering
plt.scatter(SampleData[:,0], SampleData[:,1])
plt.title("Point observations")
plt.xlabel("X")
plt.ylabel("Y")


# Euclidean Distance Calculator
def dist(dataPoint, Centroids, ax=1):
    return np.linalg.norm(dataPoint - Centroids, axis=ax)

# select centroids as per the farthest point strategy equal to the Number of Clusters we have
# first centroid is selected randomly using  following equation
# c = np.random.choice(SampleData.shape[0], 1, replace=False)
# the other centroids are selcted to be fatrthest points from the above point and other centroids.

def SelectCentroidStrategyTwo(NumberOfClusters,centroidList,c):
    centroidList = []
    pointSample = SampleData
    centroidList = pointSample[c]
    pointSample = np.delete(pointSample, c, 0)

    for c in range(1, NumberOfClusters):
        distList = [dist(pointSample[j], centroidList) for j in range(len(pointSample))] #this is an numpy array
        if (NumberOfClusters > 2):
            avgDistArr = [np.mean(distList[i]) for i in range(len(distList))]
            maxDistIndex = np.argmax(avgDistArr)
        else:
            maxDistIndex = np.argmax(distList)
        centroidList = np.vstack([centroidList, SampleData[maxDistIndex]])
        pointSample = np.delete(pointSample, maxDistIndex, 0)

    plt.scatter(centroidList[:,0], centroidList[:,1], s = 180, c = 'k' , marker = '*')
    plt.title('Initial Centroid Chosen for Number of Clusters k = %i' %(NumberOfClusters))
    # print('Centroids Selected are : ' , centroidList)
    return centroidList

# Objective Function
def ObjectiveFunc(SampleData,centroidList,clusters,meanSquaredError):
    meanSquaredError = 0
    for i in range(NumberOfClusters):
        # meanError = [(SampleData[j] - centroidList[i])  for j in range(len(SampleData)) if clusters[j] == i]
        meanSqError = [((SampleData[j] - centroidList[i])**2)  for j in range(len(SampleData)) if clusters[j] == i]
        # print('Mean Error is : ', meanError[i])
        # print('Mean Sq Error is : ', meanSqError[i])
    meanSquaredError = np.sum(meanSqError)
    ObjectiveFuncList.append(meanSquaredError)
    # print(ObjectiveFuncList)
    # print('meanSquaredError = ', meanSquaredError)
    return meanSquaredError

# Loop for number of clusters from 2 to 10
while (NumberOfClusters <= 10):
    centroidList = []
    # print('\n \n \n \n')
    # print(NumberOfClusters)
    c = np.random.choice(SampleData.shape[0], 1, replace=False)
    centroidList = SelectCentroidStrategyTwo(NumberOfClusters,centroidList,c)
    CentroidPrev = np.zeros(centroidList.shape)
    clusters = np.zeros(len(SampleData))
    ChangeInCentroidValue = dist(centroidList, CentroidPrev, None)

# begin loop over step 1 and 2 until there is no change in the centroid value
# step 1 = Classify n samples according to the nearest centroid value
# step 2 = recompute centroid values
# Finally Return centroid Values

    while(ChangeInCentroidValue != 0):
        for i in range(len(SampleData)):
            distances = dist(SampleData[i], centroidList)
            cluster = np.argmin(distances)
            clusters[i] = cluster

    # Copying previous Centroid Values in new variable
        CentroidPrev = copy.deepcopy(centroidList)

    # seperating sample data points into their respective clusters
    # and Calculating new centroids over these Clusters
        for i in range(NumberOfClusters):
            samples = [SampleData[j] for j in range(len(SampleData)) if clusters[j] == i]
            samples = np.vstack(samples)
            centroidList[i] = np.mean(samples, axis=0)

    # Calculating Change in centroid values
        ChangeInCentroidValue = dist(centroidList, CentroidPrev, None)


    # Plotting new Clusters
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    fig, ax = plt.subplots()
    for i in range(NumberOfClusters):
            samples = [SampleData[j] for j in range(len(SampleData)) if clusters[j] == i]
            samples = np.vstack(samples)
            ax.scatter(samples[:, 0], samples[:, 1], c=colors[i])

    plt.scatter(centroidList[:,0], centroidList[:,1], s = 150 , c = 'k', marker = '*')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clustering with Number of Clusters = %i' %(i+1))
    plt.show()
    meanSquaredError = 0
    ObjectiveFunc(SampleData,centroidList,clusters,meanSquaredError)
    NumberOfClusters = NumberOfClusters + 1

# print(ObjectiveFuncList)
X_numberOfClusters = [2,3,4,5,6,7,8,9,10]
plt.plot(X_numberOfClusters,ObjectiveFuncList)
plt.xlabel('Number of clusters')
plt.ylabel('Objective Function Value')
plt.title('objective function value vs. the number of clusters k')
plt.show()
