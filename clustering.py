#Eamon Collins ec3bd
#!/usr/bin/python

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def loadData(fileDj):
    data = []
    with open(fileDj, 'r') as file:
        for line in file:
            sample = []
            line = line.split()
            for i in range(len(line)):
                sample.append(float(line[i]))
            data.append(sample)

    return np.asmatrix(data)

## K-means functions

def getInitialCentroids(X, k):
    initialCentroids = []
    random.seed()
    for j in range(k):
        initialCentroids.append([random.randrange(50,80,1), random.randrange(50, 250, 1)])
    return np.asmatrix(initialCentroids)

def getDistance(pt1,pt2):
    dist = 0
    dist = math.sqrt(math.pow((pt1[0] - pt2[0]),2) + math.pow((pt1[1] - pt2[1]),2))
    return dist

def allocatePoints(X,clusters):
    n = X.shape[0]
    k = clusters.shape[0]
    labels = []
    for i in range(n):
        bestDist = getDistance((clusters[0,0],clusters[0,1]),(X[i,0],X[i,1]))
        bestCluster = 0
        for j in range(k):
            dist = getDistance((clusters[j,0],clusters[j,1]),(X[i,0],X[i,1]))
            if dist < bestDist:
                bestDist = dist
                bestCluster = j
        labels.append(bestCluster) #labels at this point are 0->k-1

    return labels

def updateCentroids(X, labels, clusters):
    k = clusters.shape[0]
    n = X.shape[0]
    newClusters = np.zeros_like(clusters)
    labelsCount = [0 for l in range(k)]
    for i in range(n):
        newClusters[labels[i], 0] += X[i,0]
        newClusters[labels[i], 1] += X[i,1]
        labelsCount[labels[i]] += 1
    for m in range(k):
        if labelsCount[m] == 0:
            newClusters[m, 0] = clusters[m,0]
            newClusters[m, 1] = clusters[m,1]
        else:
            newClusters[m, 0] /= labelsCount[m]
            newClusters[m, 1] /= labelsCount[m]
    return np.asmatrix(newClusters)


def visualizeClusters(X, labels):
    n = X.shape[0]
    x = X[:,0]
    y = X[:,1]
    use_colours = {1: "red", 2: "green", 3: "blue", 4:"yellow", 5:"cyan", 6:"orange"}
    plt.scatter(x,y, c=[use_colours[x] for x in labels])
    plt.show()


def kmeans(X, k, maxIter=1000):
    clusters = getInitialCentroids(X,k)
    iters = 0
    while(iters < maxIter):
        labels = allocatePoints(X,clusters)
        clusters = updateCentroids(X, labels, clusters)
        iters += 1
    labels = allocatePoints(X, clusters)

    return clusters, labels


def kneeFinding(X,kList):
    Objectives = []
    n = X.shape[0]
    for k in kList:
        clusters, labels = kmeans(X, k, 50)
        sumSquared = [0]*k
        for i in range(n):
            sumSquared[labels[i]] += math.pow(getDistance((X[i,0],X[i,1]),(clusters[labels[i],0],clusters[labels[i],1])),2)
        Objectives.append(sum(sumSquared))
    plt.plot(kList, Objectives)
    plt.ylabel('Objective Function')
    plt.xlabel('K')
    plt.show()
    return


def purity(X, clusters, labels):
    purities = []
    n = X.shape[0]
    k = clusters.shape[0]
    counts = np.zeros((6,6)) #row is cluster it's in, column is true label count
    for i in range(n):
        counts[labels[i], X[i,2]] += 1
    for j in range(k):
        purities.append(max(counts[j,:]) / sum(counts[j,:]))
    return purities


## GMM functions

#calculate the initial covariance matrix
#covType: diag, full
def getInitialsGMM(X,k,covType):
    X= np.asarray(X)
    p = X.shape[1] -1
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)
    initialClusters = []
    random.seed()
    for j in range(k):
        cluster = []
        for m in range(p):
            if p > 5:
                cluster.append(random.randrange(0,math.ceil(abs(X[0,m])) + 1,1))
            else:
                cluster.append(random.randrange(50,120,1))
        initialClusters.append(cluster)
    return np.asmatrix(initialClusters), np.asmatrix(covMat)


def calcLogLikelihood(X,clusters,k):
    loglikelihood = 0
    #Your code here
    return loglikelihood

#E-step
def updateEStep(X,clusters, covMat, clusterProbs, k):
    n = X.shape[0]
    p = X.shape[1]
    EMatrix = np.empty((n,k))
    for i in range(n):
        denominator = 0
        for j in range(k):
            denominator += math.exp((-1/2) * ((X[i,0:-1] - clusters[j,:]) * covMat.I * (X[i,0:-1] - clusters[j,:]).T)) * clusterProbs[j]
        for j in range(k):
            EMatrix[i,j] = math.exp((-1/2) * ((X[i,0:-1] - clusters[j,:]) * covMat.I * (X[i,0:-1] - clusters[j,:]).T)) * clusterProbs[j] / denominator
    return EMatrix

#M-step
def updateMStep(X,clusters,EMatrix):
    n = X.shape[0]
    p = X.shape[1]
    k = clusters.shape[0]
    clusterProbs = np.zeros((k))
    clusters = EMatrix.T * X[:,0:p-1]
    for i in range(n):
        clusterProbs += EMatrix[i,:]
    for j in range(k):
        clusters[j,:] /= clusterProbs[j]
    clusterProbs /= n


    return clusters, clusterProbs

def visualizeClustersGMM(X,labels,clusters,covType):
    n = X.shape[0]
    x = X[:,0]
    y = X[:,1] #just plot first two dimensions in each case
    use_colours = {1: "red", 2: "green", 3: "blue", 4:"yellow", 5:"cyan", 6:"orange"}
    plt.scatter(x,y, c=[use_colours[x] for x in labels])
    plt.show()
    return

def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM, covMat = getInitialsGMM(X,k,covType)
    clusterProbs = [1/k]*k
    labels = []
    iters = 0
    while(iters < maxIter):
        E = updateEStep(X, clustersGMM, covMat, clusterProbs, k)
        clustersGMM, clusterProbs = updateMStep(X, clustersGMM, E)
        iters+=1
    E = updateEStep(X, clustersGMM, covMat, clusterProbs, k)
    labels = labelPoints(E)
    visualizeClustersGMM(X,labels,clustersGMM,covType)
    return labels, clustersGMM

def labelPoints(E):
    labels = []
    n = E.shape[0]
    k = E.shape[1]
    for i in range(n):
        bestProb = E[i,0]
        bestInd = 0
        for j in range(k):
            if bestProb < E[i,j]:
                bestProb = E[i,j]
                bestInd = j
        labels.append(bestInd + 1)
    return labels


def purityGMM(X, clusters, labels):
    purities = []
    purities = []
    n = X.shape[0]
    k = clusters.shape[0]
    p = X.shape[1]
    counts = np.zeros((6,6)) #row is cluster it's in, column is true label count
    for i in range(n):
        counts[labels[i] - 1, X[i,p-1]] += 1
    for j in range(k):
        purities.append(max(counts[j,:]) / sum(counts[j,:]))
    return purities


def main():
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir+'/humanData.txt'
    pathDataset2 = datadir+'/audioData.txt'
    dataset1 = loadData(pathDataset1)
    dataset2 = loadData(pathDataset2)
    MAX_ITER = 200

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    clusters, labels = kmeans(dataset1, 2, maxIter=MAX_ITER)
    visualizeClusters(dataset1, labels)
    print("Purity values for each cluster: " + repr(purity(dataset1,clusters, labels)))

    #Q7
    #labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'diag', MAX_ITER)
    #labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full', MAX_ITER)

    #Q8
    labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag', MAX_ITER)
    #labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full', MAX_ITER)

    #Q9
    #purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    #print(purities11)
    #purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    #print(purities12)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    print(purities21)
    #purities22 = purityGMM(dataset2, clustersGMM22, labels22)
    #print(purities22)

if __name__ == "__main__":
    main()