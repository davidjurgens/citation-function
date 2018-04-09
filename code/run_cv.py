def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import unicodecsv
import json
import datetime
from datetime import date, timedelta as td
import time
import re
import numpy
import numpy as np
from numpy import histogram
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
import csv
import sys
import os

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import LogisticRegressionCV
#import theano
from sklearn import svm
import copy
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import LeaveOneOut
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
import random
from sklearn.metrics import *
from collections import Counter
from sklearn import preprocessing
import sys

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import *
from sklearn.dummy import DummyClassifier


csv.field_size_limit(sys.maxsize)

if len(sys.argv) != 2:
    print 'usage: run_cv.py feature-dir/'
    sys.exit(1)

feature_dir = sys.argv[1]
make_gold_network = False

if make_gold_network:
    fw = open("gold-standard-network.tsv","w")

def get_section(vec):
    if 'InAbstract' in feature_to_index and vec[feature_to_index['InAbstract']] == 1:
        return 'Abstract'
    elif vec[feature_to_index['InIntroduction']] == 1:
        return 'Introduction'
    elif 'InMotivation' in feature_to_index and vec[feature_to_index['InMotivation']] == 1:
        return 'Motivation'
    elif vec[feature_to_index['InRelatedWork']] == 1:
        return 'Related_Work'
    elif vec[feature_to_index['InMethology']] == 1:
        return 'Methodology'
    elif vec[feature_to_index['InEvaluation']] == 1:
        return 'Evaluation'
    elif 'InResults' in feature_to_index and vec[feature_to_index['InResults']] == 1:
        return 'Results'
    elif 'InDiscussion' in feature_to_index and vec[feature_to_index['InDiscussion']] == 1:
        return 'Discussion'
    elif vec[feature_to_index['InConclusion']] == 1:
        return 'Conclusion'
    elif 'InAcknowledgements' in feature_to_index and vec[feature_to_index['InAcknowledgements']] == 1:
        return 'Acknowledgements'
    elif 'InReferences' in feature_to_index and vec[feature_to_index['InReferences']] == 1:
        return 'References'
    else:
        return 'Other_Section'

feature_to_index = {}
with open(feature_dir + 'feature-indices.tsv') as f:
    for line in f:
        cols = line.strip().split('\t');
        feature = cols[0]
        index = int(cols[1])
        feature_to_index[feature] = index


x = []
y = []
cited_paper_ids = []
cited_paper_sections = []
citing_paper_ids = []
paper_indices = []

count = 0

#files = [fname for fname in os.listdir(feature_dir)]
#random.shuffle(files)

for fname in os.listdir(feature_dir):
    if '.ftr' not in fname:
        print 'skipping ' + fname
        continue
    f = open(feature_dir + fname, "r")
    citing_paper_id = fname[0:8]

    #print feature_dir + fname
    rdr = csv.reader(f, delimiter="\t")
    to_append = []
    for cols in rdr:
        if len(cols) < 10:
            continue


        cite_function = cols[1].split('-')[0].strip()
        if cite_function == 'Unknown':
            continue

        # Merge the "Prior" and "Continues" classes
        if cite_function == 'Prior':
            cite_function = 'Extends'

        # Merge the "Compares" and "Contrasts" classes
        if cite_function.startswith("Co"):
            cite_function = 'CompareOrContrast'

        cite_centrality = cols[2].strip()

        # Distinguish between Essential-CoCo and Positioning-CoCo
        if cite_function.startswith("Co"):
            #cite_function = cite_centrality + '-' + cite_function
            pass

        class_ = cite_function
        # print class_

        feature_vec = map(float, cols[4:])
        x.append(feature_vec)

        y.append(class_) # 9 Classes, 

        cited_paper_ids.append(cols[0])
        if make_gold_network:
            cited_paper_sections.append(get_section(feature_vec))
        citing_paper_ids.append(citing_paper_id)

        # print citing_paper_id, ' -> ', cols[0]
        
        to_append.append(count)
        count += 1
    if len(to_append) > 0:
        paper_indices.append(to_append)


x = numpy.array(x)
y = numpy.array(y)
print x.shape, y.shape, len(paper_indices)

print x.shape[1]/2

######################################
# Data loaded -- now on to classification

# ALL CLASS LABELS

TOTAL_PAPERS = len(paper_indices)

accscores = []

x_train = x[:,:]

sections = x[:,1]

y_overall = numpy.array([0.0]*len(y))

ALL_Y_PREDICTED = []
ALL_Y_REAL = []

# pe_debug = open('cv.cite-func.all-features.tsv', 'w')

#for idx in range(10):
for idx in range(TOTAL_PAPERS):
    test = paper_indices[idx]
    train = list(set([item for sublist in paper_indices for item in sublist]) - set((test)))
            
    train_x = x_train[train][:,1:]    
    test_x = x_train[test][:,1:]  
        
    train_y = y[train]
    test_y = y[test]   

    to_ids = numpy.array(cited_paper_ids)[test]
    from_ids = numpy.array(citing_paper_ids)[test]

    if make_gold_network:
        sections_ids = numpy.array(cited_paper_sections)[test]
        

    #clf = DummyClassifier(strategy='uniform') #strategy='constant', constant='Essential')
    #clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

                                     #max_features=0.5, 

    #clf = ExtraTreesClassifier(n_estimators=5000, \
        
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
                                     #clf = KNeighborsClassifier(n_neighbors=3)

                                     
    #clf = GaussianNB()
    #clf = svm.LinearSVC()
    # clf = DummyClassifier(strategy='uniform')
    
    #skb = SelectKBest(f_classif, k=50)
    #Train_x = skb.fit_transform(train_x, train_y)
    clf.fit(train_x, train_y)
    

    predicted_y = clf.predict(test_x)

    #for i, py in enumerate(predicted_y):
    #    pe_debug.write('\t'.join(map(str, [from_ids[i], to_ids[i], py, test_y[i]])) + '\n')
    #pe_debug.flush()

    #predicted_y = clf.predict(skb.transform(test_x))
    #print predicted_y
    accscores.append(accuracy_score(test_y, predicted_y))

    # Srijan -- how do i print the network here?

    if make_gold_network:
        for i in range(len(predicted_y)):
            fw.write('%s\t%s\t%s\t%s\t%s\n' % (from_ids[i], to_ids[i], sections_ids[i], test_y[i], predicted_y[i]))
    
    ALL_Y_REAL.extend(test_y)
    ALL_Y_PREDICTED.extend(predicted_y)

    labels = set(ALL_Y_REAL)
    labels |= set(ALL_Y_PREDICTED)
    labels = list(labels)

    #print accuracy_score(test_y, predicted_y)
    
    lb = preprocessing.LabelEncoder()
    prec = -1.0 #precision_score(lb.fit_transform(ALL_Y_REAL), lb.transform(ALL_Y_PREDICTED))
    rec = -1.0 #recall_score(lb.transform(ALL_Y_REAL), lb.transform(ALL_Y_PREDICTED))

    macro_f1 = f1_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='macro')
    micro_f1 = f1_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='micro')
    macro_p = precision_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='macro')
    micro_p = precision_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='micro')
    macro_r = recall_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='macro')
    micro_r = recall_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='micro')

    print 'Running accuracy after %d papers, %f macro F1 (P: %f, R: %f), %f micro F1 (P: %f, R: %f)' \
        % (idx+1, macro_f1, macro_p, macro_r, micro_f1, micro_p, micro_r)
    
    print labels
    print sklearn.metrics.confusion_matrix(ALL_Y_REAL, ALL_Y_PREDICTED, \
        labels=labels)
    print '---------------------\n\n'

# pe_debug.close()
  
if make_gold_network:
    fw.close()
print "---"
print clf
print "ACCURACY", numpy.mean(accscores)

print "Micro accuracy score", accuracy_score(ALL_Y_REAL, ALL_Y_PREDICTED)

print ' '.join(['Background', 'Motivation', 'Prior', 'Uses',  'Extends', 'CompareOrContrast', 'Future'])
#print ' '.join(['Positional', 'Essential'])


print sklearn.metrics.confusion_matrix(ALL_Y_REAL, ALL_Y_PREDICTED, \
    labels=['Background', 'Motivation', 'Prior', 'Uses',  'Extends', 'CompareOrContrast', 'Future'])

#print sklearn.metrics.confusion_matrix(ALL_Y_REAL, ALL_Y_PREDICTED, \
#    labels=['Positional', 'Essential'])
    
