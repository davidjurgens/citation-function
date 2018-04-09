import unicodecsv
import json
import datetime
from datetime import date, timedelta as td
import time
import re
import numpy
import numpy as np
import matplotlib
from numpy import histogram
from matplotlib import rcParams
import cPickle
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from nltk.tokenize.punkt import PunktSentenceTokenizer as sentTokenizer
from os import listdir
import math
from collections import defaultdict
import pandas as pd
import sklearn
import random
import matplotlib
import csv
import sys
import os
import pickle
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import LogisticRegressionCV
#import theano
from sklearn import svm
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
from sklearn.metrics import *
from collections import Counter
from sklearn import preprocessing
import sys

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


from imblearn import over_sampling as osam
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced

from imblearn.combine import SMOTETomek

from imblearn.combine import SMOTEENN


csv.field_size_limit(sys.maxsize)

if len(sys.argv) != 3:
    print 'usage: python classify-papers.py input-dir/ classifier.pickle'
    quit(1)


feature_dir = sys.argv[1]
outfile = sys.argv[2]

#if len(sys.argv[3]) > 3:


feature_to_index = {}
with open(feature_dir + 'feature-indices.tsv') as f:
    for line in f:
        cols = line.strip().split('\t');
        feature = cols[0]
        index = int(cols[1])
        feature_to_index[feature] = index


print 'Loading topic vectors'
citation_id_to_topic_vecs = {}

num_topics = 100

with open('../working-files/topics3/%d-topics/citance.doc-topics.txt' % num_topics) as citance_topic_f:
    with open('../working-files/topics3/%d-topics/extended-citance.doc-topics.txt' % num_topics) as extended_topic_f:
        with open('../working-files/context-ids.tsv') as f:
            for line_no, line in enumerate(f):
                cols = line[:-1].split('\t')
                citation_id = cols[2]

                citance_topic_line = citance_topic_f.readline()
                extended_topic_line = extended_topic_f.readline()

                citance_topic_vec = np.array(map(float, citance_topic_line[:-1].split('\t')[2:]))
                extended_topic_vec = np.array(map(float, extended_topic_line[:-1].split('\t')[2:]))
                
                citation_id_to_topic_vecs[citation_id] = (citance_topic_vec, extended_topic_vec)
                
                if (line_no+1) % 10000 == 0:
                    print 'loaded %d lines' % (line_no+1)
                if (line_no+1) % 100000 == 0:
                    break

print 'Loaded %d topic vectors' % (len(citation_id_to_topic_vecs))


x = []
y = []
paper_indices = []
count = 0
for fname in os.listdir(feature_dir):
    if '.ftr' not in fname:
        print 'skipping ' + fname
        continue
    f = open(feature_dir + fname, "r")
    #print feature_dir + fname
    rdr = csv.reader(f, delimiter="\t")
    to_append = []
    for cols in rdr:
        if len(cols) < 10:
            continue


        cite_function = cols[1].split('-')[0].strip()
        citation_id = cols[2].strip()
        cite_type = cols[3].strip()

        # print citation_id
        if citation_id not in citation_id_to_topic_vecs:
            print 'skipping!'
            continue

        if cite_function == 'Unknown':
            continue


        # Merge the "Prior" and "Extends" classes
        if cite_function == 'Prior':
            cite_function = 'Extends'

        # Merge the "Compares" and "Contrasts" classes
        if cite_function.startswith("Co"):
            cite_function = 'CompareOrContrast'


        # Distinguish between Essential-CoCo and Positioning-CoCo
        #if cite_function.startswith("Co"):
        #    cite_function = cite_type + '-' + cite_function
        #    #pass

        class_ = cite_function
        # print class_

        feature_vec = map(float, cols[4:])
        topic_vecs = citation_id_to_topic_vecs[citation_id]
        feature_vec = np.concatenate([feature_vec, topic_vecs[0], topic_vecs[1]])


        x.append(feature_vec)
        y.append(class_) # 9 Classes, 
        
        to_append.append(count)
        count += 1
    paper_indices.append(to_append)


_c = Counter(y)
class_weights = {}
for xx, cc in _c.iteritems():
    weight = len(y) / (len(_c) * float(cc))
    if xx == 'CompareOrContrast':
        weight += (weight/2)
    print 'count:', xx, cc, weight
    class_weights[xx] = weight

class_weights2 = {
    'Motivation': 10,
    'Future': 10,
    'Extends': 10,
    'Background': 1,
    'Uses': 2,
    'CompareOrContrast': 5
}

class_weights3 = {
    'Motivation': 3,
    'Future': 3,
    'Extends': 3,
    'Background': 1,
    'Uses': 1.5,
    'CompareOrContrast': 2
}


#n_samples / (n_classes * np.bincount(y))

x = numpy.array(x)
y = numpy.array(y)
print x.shape, y.shape, len(paper_indices)

######################################
# Training data loaded -- build the classifier
    
#clf = RandomForestClassifier(n_estimators=500, \
#                                 max_features='auto', n_jobs=-1, \
#                                 min_samples_leaf=1, warm_start=True)

# clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, min_samples_leaf=2, class_weight='balanced_subsample')

#../working-files/grid-search.50-topic/RandomForestClassifier.n_estimators-1000.max_features-0.1.min_samples_leaf-5.class_weight-balanced_subsample.warm_start-True.smote-True.merge_cc-False.log:Running accuracy after 271 papers, 0.481234 macro F1, 0.605769 micro F1

# ../working-files/grid-search.50-topic/RandomForestClassifier.n_estimators-1000.max_features-0.25.min_samples_leaf-3.class_weight-balanced_subsample.warm_start-True.smote-True.merge_cc-False.log:Running accuracy after 271 papers, 0.481421 macro F1, 0.617308 micro F1

# clf = RandomForestClassifier(n_estimators=2500, n_jobs=-1, min_samples_leaf=3, class_weight='balanced_subsample', max_features=0.25)



#  n_estimators-2500.max_features-sqrt.min_samples_leaf-5.class_weight-balanced.warm_start-False.smote-True.smote-nn-2.tomek-False.merge_cc-True.log:Running accuracy after 271 papers, 0.510822 macro F1, 0.609615 micro F1


#../working-files/grid-search.50-topic.2/RandomForestClassifier.n_estimators-2500.max_features-0.1.min_samples_leaf-7.class_weight-balanced_subsample.warm_start-False.smote-True.smote-nn-10.tomek-True.merge_cc-True.log:Running accuracy after 271 papers, 0.534674 macro F1, 0.615000 micro F1

# ../working-files/grid-search.50-topic.2/RandomForestClassifier.n_estimators-2500.max_features-0.25.min_samples_leaf-5.class_weight-balanced_subsample.warm_start-False.smote-False.smote-nn-5.tomek-False.merge_cc-True.log:Running accuracy after 271 papers, 0.529983 macro F1, 0.625769 micro F1


clf = RandomForestClassifier(n_estimators=2500, n_jobs=-1, min_samples_leaf=6,
                             max_features=0.10, class_weight=class_weights,
                             warm_start=False)

smote = osam.SMOTE(k_neighbors=5, ratio='auto')
#clf = pl.make_pipeline(SMOTETomek(smote=smote), clf)
clf = pl.make_pipeline(smote, clf)

print 'Training the classifer'
clf.fit(x, y)

#joblib.dump(clf, outfile, compress=1) 

with open(outfile, 'wb') as outf:
    pickle.dump(clf, outf)
