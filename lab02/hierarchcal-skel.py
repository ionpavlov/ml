# Tudor Berariu, 2016

from sys import argv
from zipfile import ZipFile
from random import randint

import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import matplotlib.markers
import math
from mpl_toolkits.mplot3d import Axes3D

def getArchive():
    archive_url = "http://www.uni-marburg.de/fb12/datenbionik/downloads/FCPS"
    local_archive = "FCPS.zip"
    from os import path
    if not path.isfile(local_archive):
        import urllib
        print("downloading...")
        urllib.urlretrieve(archive_url, filename=local_archive)
        assert(path.isfile(local_archive))
        print("got the archive")
    return ZipFile(local_archive)

def getDataSet(archive, dataSetName):
    path = "FCPS/01FCPSdata/" + dataSetName

    lrnFile = path + ".lrn"
    with archive.open(lrnFile, "r") as f:                       # open .lrn file
        N = int(f.readline().decode("UTF-8").split()[1])    # number of examples
        D = int(f.readline().decode("UTF-8").split()[1]) - 1 # number of columns
        f.readline()                                     # skip the useless line
        f.readline()                                       # skip columns' names
        Xs = np.zeros([N, D])
        for i in range(N):
            data = f.readline().decode("UTF-8").strip().split("\t")
            assert(len(data) == (D+1))                              # check line
            assert(int(data[0]) == (i + 1))
            Xs[i] = np.array(list(map(float, data[1:])))

    clsFile = path + ".cls"
    with archive.open(clsFile, "r") as f:                        # open.cls file
        labels = np.zeros(N).astype("uint")

        line = f.readline().decode("UTF-8")
        while line.startswith("%"):                                # skip header
            line = f.readline().decode("UTF-8")

        i = 0
        while line and i < N:
            data = line.strip().split("\t")
            assert(len(data) == 2)
            assert(int(data[0]) == (i + 1))
            labels[i] = int(data[1])
            line = f.readline().decode("UTF-8")
            i = i + 1

        assert(i == N)

    return Xs, labels                          # return data and correct classes


def dummy(Xs):
    (N, D) = Xs.shape
    Z = np.zeros((N-1, 4))
    lastIndex = 0
    for i in range(N-1):
        Z[i,0] = lastIndex
        Z[i,1] = i+1
        Z[i,2] = 0.1 + i
        Z[i,3] = i+2
        lastIndex = N+i
    return Z

def euclidianDistance(p1, p2):
    d = 0
    for j in range(p1.size):
        d = d + (p1[j] - p2[j]) * (p1[j] - p2[j])

    return d

def calculateMatrixDistance(Xs, N):
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                D[i,j] = None
            else:
                D[i,j] = euclidianDistance(Xs[i], Xs[j])

    return D

def singleLinkageDistance(D, cluster1, cluster2):
    min_d = None
    for i in cluster1:
        for j in cluster2:
            d = D[i,j]
            if not min_d or d < min_d:
                min_d = d

    return min_d

def completeLinkageDistance(D, cluster1, cluster2):
    max_d = None
    for i in cluster1:
        for j in cluster2:
            d = D[i,j]
            if not max_d or d > max_d:
                max_d = d

    return max_d

def groupAverageDistance(D, cluster1, cluster2):
    Sum = 0
    for i in cluster1:
        for j in cluster2:
            d = D[i, j]
            Sum = Sum + d

    return Sum/(len(cluster1)*len(cluster2))

def findClustersToUnify(D, dict, distanceType):
    min_d = None
    for key1 in dict.keys():
        for key2 in dict.keys():
            d = 0
            if key1 != key2:
                c1 = dict[key1]
                c2 = dict[key2]
                if distanceType == "SINGLE":
                    d = singleLinkageDistance(D, c1, c2)
                elif distanceType == "COMPLETE":
                    d = completeLinkageDistance(D, c1, c2)
                elif distanceType == "GROUP":
                    d = groupAverageDistance(D, c1, c2)
                if not min_d or d < min_d:
                    cluster1 = key1
                    cluster2 = key2
                    min_d = d

    return cluster1, cluster2, min_d

def unifyClusters(Xs, Z, dict, N, distanceType):
    D = np.zeros((N, N))
    D = calculateMatrixDistance(Xs, N)

    for i in range(N-1):
        cluster1, cluster2, min_d = findClustersToUnify(D, dict, distanceType)
        dict[N+i] = dict[cluster1] + dict[cluster2]
        del dict[cluster1]
        del dict[cluster2]
        Z[i, 0] = cluster1
        Z[i, 1] = cluster2
        Z[i, 2] = min_d
        Z[i, 3] = len(dict[N+i])
    return Z


def singleLinkage(Xs):
    N = len(Xs)
    Z = np.zeros((N-1, 4))
    dict = {}

    #init dictionary
    for i in range(N):
        dict[i] = [i]
    Z = unifyClusters(Xs, Z, dict, N, "SINGLE")
    return Z


def completeLinkage(Xs):
    N = len(Xs)
    Z = np.zeros((N - 1, 4))
    dict = {}

    # init dictionary
    for i in range(N):
        dict[i] = [i]
    Z = unifyClusters(Xs, Z, dict, N, "COMPLETE")
    return Z

def groupAverageLinkage(Xs):
    N = len(Xs)
    Z = np.zeros((N - 1, 4))
    dict = {}

    # init dictionary
    for i in range(N):
        dict[i] = [i]
    Z = unifyClusters(Xs, Z, dict, N, "GROUP")
    return Z


def extractClusters(Xs, Z, clusterNumber):
    (N, Dim) = Xs.shape
    assert(Z.shape == (N-1, 4))
    clusters = np.zeros(N).astype("uint")
    #facem merge pentru crearea clusterelor
    dict = {}
    # init dictionary
    for i in range(N):
        dict[i] = [i]
    limit = N - clusterNumber
    for i in range(limit):
        cluster1 = int(Z[i, 0])
        cluster2 = int(Z[i, 1])
        dict[N+i] = dict[cluster1] + dict[cluster2]
        del dict[cluster1]
        del dict[cluster2]

    k = -1
    for key,value in dict.items():
        k = k + 1
        for i in value:
            clusters[i] = k

    return len(dict), clusters

def randIndex(clusters, labels):
    assert(labels.size == clusters.size)
    N = clusters.size

    a = 0.0
    b = 0.0

    for (i, j) in [(i,j) for i in range(N) for j in range(i+1, N) if i < j]:
        if ((clusters[i] == clusters[j]) and (labels[i] == labels[j]) or
            (clusters[i] != clusters[j]) and (labels[i] != labels[j])):
            a = a + 1
        b = b + 1

    return float(a) / float(b)

def plot(Xs, labels, K, clusters):
    labelsNo = np.max(labels)
    markers = []                                     # get the different markers
    while len(markers) < labelsNo:
        markers.extend(list(matplotlib.markers.MarkerStyle.filled_markers))
    colors = plt.cm.rainbow(np.linspace(0, 1, K+1))

    if Xs.shape[1] == 2:
        x = Xs[:,0]
        y = Xs[:,1]
        for (_x, _y, _c, _l) in zip(x, y, clusters, labels):
            plt.scatter(_x, _y, s=200, c=colors[_c], marker=markers[_l])
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:,0]
        y = Xs[:,1]
        z = Xs[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=colors[_c], marker=markers[_l])
        plt.show()
    else:
        for i in range(N1):
            print(i, ": ", clusters[i], " ~ ", labels[i])


if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: " + argv[0] + " dataset_name")
        exit()

    Xs, labels = getDataSet(getArchive(), argv[1])    # Xs is NxD, labels is Nx1

    Z = singleLinkage(Xs)
    fig1 = plt.figure()
    dn = hierarchy.dendrogram(Z)
    fig1.show()

    Z = completeLinkage(Xs)
    fig2 = plt.figure()
    dn = hierarchy.dendrogram(Z)
    fig2.show()

    Z = groupAverageLinkage(Xs)
    fig3 = plt.figure()
    dn = hierarchy.dendrogram(Z)
    fig3.show()

    K, clusters = extractClusters(Xs, Z, 6)
    print("randIndex: ", randIndex(clusters, labels))

    plot(Xs, labels, K, clusters)
