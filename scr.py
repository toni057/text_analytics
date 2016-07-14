# -*- coding: utf-8 -*-
"""
Created on Sat Jul 09 20:33:21 2016

@author: toni
"""

#%% set working directory


from os import chdir

chdir('C:\\Users\\toni\\Desktop\\Clustering and retrieval\\W2\\people_wiki\\')


#%% load libraries

import pandas as pd
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices



#%% load data



df = pd.read_csv('people_wiki.csv')



#%% remove all words with digits

import re

tokens2 = list()

for i in range(df.shape[0]):
    df.ix[i,2] = re.sub(r'\w*\d\w*', '', df.ix[i,2]).strip()



#%% tokenize

from nltk.tokenize import word_tokenize

tokens = list()

for text in df.text[:10]:
    tokens.append(word_tokenize(text))


#%% convert to lowercase


lower = list()

for t in tokens:
    lower.append([w.lower() for w in t])




#%% remove stopwords

from nltk.corpus import stopwords

# stopwords
stop = set(stopwords.words('english'))

# tokens with removed stopwords
tokens2 = list()

for t in lower:
    tokens2.append([w for w in t if w not in stop])


#del tokens


#%% stemming 

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

stemmed = list()

for t in tokens2[:10]:
    stemmed.append([stemmer.stem(w) for w in t])
    

#%%

# flatten a list and make a set

import itertools

vocabulary = set(itertools.chain.from_iterable(stemmed))







#%% creating a sparse design matrix - term document matrix

indptr = [0]
indices = []
data = []
vocabulary = {}

for t in tokens:
    for term in t:
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))


m = csr_matrix((data, indices, indptr), dtype=int).toarray()


