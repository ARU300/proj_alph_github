import load
import pprint
import datetime
from pymongo import MongoClient
import pymongo
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from numpy import arange
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

import scipy
import pandas
import numpy
import re
import nose
import random
import string
import time
import threading

from pathlib import Path

filename = 'text.txt'  # load data
file = open(filename, 'rt')  # load data
tbtext = file.read()  # load data
file.close()  # load 

def text_normal(text):
    print('Tokenisation Start')
    text = str(text).lower()  # text to lower case
    #text = lowercase(str(text))
    table = str.maketrans('', '', string.punctuation)  # remove punctuation
    tokens = word_tokenize(text)  # tokenisation
    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    words = [word for word in stripped if word.isalpha()] # remove special characters
    stop_words = set(stopwords.words('english'))  # remove stopwords
    words = [w for w in words if not w in stop_words]  # remove stopwords
    print(words[:250])
    print('Tokenisation End')

    print('Lemmetization Start')
    lemma = wordnet.WordNetLemmatizer  # start lemmatization
    tags_list = pos_tag(tokens, tagset=None)  # parts of speech
    lemma_words = []  # empty list
    for token, pos_token in tags_list:
        if pos_token.startswith('V'):  # verb
            pos_val = 'v'
        elif pos_token.startswith('J'):  # Adjective
            pos_val = 'j'
        elif pos_token.startswith('R'):  # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n'  # Noun
        lemma_token = lemma.lemmatize(
            token, pos_token)  # performing lemmatization
        lemma_words.append(lemma_token)  # appending the lemmatized tokens
        print(text_normal(tbtext[:500]))
        print('Lemmatization End')
    return " ".join(lemma_words)  # returns the lemmatized tokens as a sentence


def freqmatrix(words):
    frequency_matrix = {}
    for sent in tokens:
        freq_table = {}
        for word in words:
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


text_normal(tbtext)
freqmatrix(words)
print(freqmatrix(words))
print('Frequency Matrix Created')