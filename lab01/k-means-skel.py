# Tudor Berariu, 2016

from sys import argv
from zipfile import ZipFile
from random import randint


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers
import copy
import random
import math
import operator
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

def kMeans(K, Xs):
    (N, D) = Xs.shape
    #D - space dimension
    #N = points number
    centroids = np.zeros((K, D))
    clusters = np.zeros(N).astype("uint")       # id of cluster for each example
    # TODO: Cerinta 1
    #select random centroids
    for k in range(K):
        r = random.randint(0, K-1)
        centroids[k] = Xs[r]

    while True:
        oldclusters = clusters.copy()
        for i in range(N):
            x = Xs[i]
            min_d = None

            for k in range(K):
                d = 0
                for j in range(x.size):
                    d = d + (x[j] - centroids[k][j])*(x[j] - centroids[k][j])

                d = math.sqrt(d)

                if not min_d or d < min_d:
                    min_d = d
                    clusters[i] = k

        #calculate new centroid
        for k in range(K):
            nr = 0
            vect = np.zeros(D)

            for i in range(N):
                if clusters[i] == k:
                    nr = nr + 1

                    for j in range(D):
                        vect[j] = vect[j] + Xs[i][j]

            if nr != 0:
                for j in range(D):
                    centroids[k][j] = operator.truediv(vect[j], nr)

        # check termination
        if np.all(oldclusters == clusters):
            break

    return clusters, centroids

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
            plt.scatter(_x, _y, s=500, c=colors[_c], marker=markers[_l])
        plt.scatter(centroids[:,0], centroids[:, 1],
                    s=800, c=colors[K], marker=markers[labelsNo]
        )
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:,0]
        y = Xs[:,1]
        z = Xs[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=colors[_c], marker=markers[_l])
        ax.scatter(centroids[:,0], centroids[:, 1], centroids[:, 2],
                    s=400, c=colors[K], marker=markers[labelsNo]
        )
        plt.show()
    else:
        for i in range(N1):
            print(i, ": ", clusters[i], " ~ ", labels[i])

if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: " + argv[0] + " dataset_name K")
        exit()
    Xs, labels = getDataSet(getArchive(), argv[1])    # Xs is NxD, labels is Nx1
    K = int(argv[2])                                # K is the number of clusters

    clusters, centroids = kMeans(K, Xs)
    print("randIndex: ", randIndex(clusters, labels))
    print(centroids)
    plot(Xs, labels, K, clusters)
