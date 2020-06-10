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
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from numpy import arange
from numpy import unique
from numpy import where
from pathlib import Path

# define dataset
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


def cluster_plot(type, classify):
    yhat = model.fit_predict(classify)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    pyplot.show()


model = DBSCAN(eps=0.5, min_samples=50, n_jobs=1)
cluster_plot(type=model, classify=X)

model = KMeans(n_clusters=4)
cluster_plot(type=model, classify=X)

quit()
