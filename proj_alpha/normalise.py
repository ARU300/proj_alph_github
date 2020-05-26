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

#tbtext = pd.read_csv('tbtext_E6oV3lV.csv')



def text_normal(text):
    print('Tokenisation Start')
    text = str(text).lower()  # text to lower case
    #text = lowercase(str(text))
    table = str.maketrans('', '', string.punctuation)  # remove punctuation
    tokens = word_tokenize(text)  # tokenisation
    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    # remove special characters
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))  # remove stopwords
    words = [w for w in words if not w in stop_words]  # remove stopwords
    #freq = pandas.Series(' '.join(tbtext['text']).split()).value_counts()[:10] #remove common words
    #freq = list(freq.index)
    #tbtext['text'] = tbtext['text'].apply(
    #lambda x: " ".join(x for x in x.split() if x not in freq))
    #tbtext['text'].head()
    #freq = pandas.Series(' '.join(tbtext['text']).split()).value_counts()[-10:] #remove rare words
    #freq = list(freq.index)
    #tbtext['text'] = tbtext['text'].apply(
    #lambda x: " ".join(x for x in x.split() if x not in freq))
    #tbtext['text'].head()
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
    return " ".join(lemma_words)  # returns the lemmatized tokens as a sentence


text_normal(tbtext)
print(text_normal(tbtext[:500]))
print('Lemmatization End')

colors = 'rgbcmyk'  # red, green, blue, cyan, magenta, yellow, black


def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind+c*width, counts[categories[c]], width,
                          color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind+width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pyplot.ylabel('Frequency')
    pyplot.title('Frequency of Six Modal Verbs by Genre')
    pyplot.show()


#genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
#modals = ['can', 'could', 'may', 'might', 'must', 'will']
#cfdist = nltk.ConditionalFreqDist(
#        (genre, word)
#        for genre in genres
#        for word in tbtext(categories=genre)
#        if word in modals)
#counts = {}
#for genre in genres:
#    counts[genre] = [cfdist[genre][word] for word in modals]
#bar_chart(genres, modals, counts)
