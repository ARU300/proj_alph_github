import load
import pprint
import datetime
import pymongo
from pymongo import MongoClient
import scipy

import re
import nose
import random
import string
import time
import threading

import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import datetime
from sklearn import datasets
from numpy import arange
from numpy import unique
from numpy import where
from pathlib import Path

# define dataset
X, y = make_blobs(n_samples=2000, centers=4, cluster_std=0.70, shuffle=False, random_state=1)


def cluster_plot(model_type, classify):
    yhat = model.fit_predict(classify)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    return pyplot


inp = input('What clustering method do you want to test?\n')

if str(inp) == 'dbscan':
    begin_time = datetime.datetime.now()
    model = DBSCAN(eps=0.4, min_samples=5, n_jobs=1)
    cluster_plot(model_type=model, classify=X)
    labels = model.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    pyplot.title('DB-SCAN')
    pyplot.show()
    print('Execution time was {}'.format(datetime.datetime.now() - begin_time))
elif str(inp) == 'kmeans':
    begin_time = datetime.datetime.now()
    model = KMeans(n_clusters=4)
    cluster_plot(model_type=model, classify=X)
    pyplot.title('K Means')
    pyplot.show()
    print('Execution time was {}'.format(datetime.datetime.now() - begin_time))
elif str(inp) == 'all':
    begin_time = datetime.datetime.now()
    model = DBSCAN(eps=0.35, min_samples=7)
    cluster_plot(model_type=model, classify=X)
    labels = model.labels_
    cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_num -= 1
    noise_num = list(labels).count(-1)
    print(noise_num)
    p = []
    i = 0
    for points in X:
        p.append(points)
        i += 1

    ka = []
    j = 0
    for k in labels:
        ka.append(k)
        j += 1

    j -= 1
    delete = []
    while j != 0:
        ka_value = ka[j]
        if ka_value == -1:
            print(ka_value)
            x = p[j]
            print(x)
            #x.remove()
        j -= 1


    pyplot.title('DB-SCAN')
    pyplot.show()

    model = KMeans(n_clusters=cluster_num)
    cluster_plot(model_type=model, classify=X)
    pyplot.title('K Means')
    pyplot.show()
    print('Execution time was {}'.format(datetime.datetime.now() - begin_time))

quit()
