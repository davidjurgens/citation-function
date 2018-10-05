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
import sklearn
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
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import *
from sklearn.dummy import DummyClassifier

from imblearn import over_sampling as osam
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced

from imblearn.combine import SMOTETomek

from imblearn.combine import SMOTEENN

csv.field_size_limit(sys.maxsize)


if len(sys.argv) < 2:
    print 'usage: run_cv.py feature-dir/'
    sys.exit(1)

feature_dir = sys.argv[1]

print 'Loading topic vectors'
citation_id_to_topic_vecs = {}
with open('../working-dir/topics/citance.doc-topics.txt') as citance_topic_f:
    with open('../working-dir/topics/extended-citance.doc-topics.txt') as extended_topic_f:
        with open('../working-dir/topics/context-ids.tsv') as f:
            for line_no, line in enumerate(f):
                cols = line[:-1].split('\t')                             
                

                citation_id = cols[2]

                citance_topic_line = citance_topic_f.readline()
                extended_topic_line = extended_topic_f.readline()

                t_citation_id = citance_topic_line[:-1].split('\t')[1]
                if t_citation_id != citation_id.replace('.', ''):
                    raise Exception('BAD TOPICS: %s != %s' % (citation_id, t_citation_id))

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
cited_paper_ids = []
cited_paper_sections = []
citing_paper_ids = []
paper_indices = []

count = 0

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

        class_ = cite_function

        feature_vec = map(float, cols[4:])
        topic_vecs = citation_id_to_topic_vecs[citation_id]
        feature_vec = np.concatenate([feature_vec, topic_vecs[0], topic_vecs[1]])
        #feature_vec = np.concatenate([topic_vecs[0], topic_vecs[1]])



        x.append(feature_vec)

        y.append(class_) # 9 Classes, 

        cited_paper_ids.append(cols[0])
        citing_paper_ids.append(citing_paper_id)

        # print citing_paper_id, ' -> ', cols[0]
        
        to_append.append(count)
        count += 1
    if len(to_append) > 0:
        paper_indices.append(to_append)


print set(y)

x = numpy.array(x)
y = numpy.array(y)



print x.shape, y.shape, len(paper_indices)

# print x.shape[1]/2

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

print 'beginning training'

#for idx in range(10):
for idx in range(TOTAL_PAPERS):
    #print 'holding out paper', idx
    test = paper_indices[idx]
    train = list(set([item for sublist in paper_indices for item in sublist]) - set((test)))

    #print 'splitting into train/test'
    train_x = x_train[train][:,1:]    
    test_x = x_train[test][:,1:]  
        
    train_y = y[train]
    test_y = y[test]   

    #print 'getting IDs'
    to_ids = numpy.array(cited_paper_ids)[test]
    from_ids = numpy.array(citing_paper_ids)[test]      
    
    # clf = DummyClassifier(strategy='uniform')
    # clf = DummyClassifier(strategy='constant', constant='Background')
            
    clf = RandomForestClassifier(n_estimators=2500, n_jobs=-1, min_samples_leaf=5, random_state=1234)
    
    smote = osam.SMOTE(k_neighbors=5, ratio='auto')
    clf = pl.make_pipeline(smote, clf)
    clf.fit(train_x, train_y)
    

    predicted_y = clf.predict(test_x)
    accscores.append(accuracy_score(test_y, predicted_y))
    
    ALL_Y_REAL.extend(test_y)
    ALL_Y_PREDICTED.extend(predicted_y)

    labels = set(ALL_Y_REAL)
    labels |= set(ALL_Y_PREDICTED)
    labels = list(labels)
   

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
  
print "---"
print clf
#print "ACCURACY", numpy.mean(accscores)

print "Macro F1: score", f1_score(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels, average='macro')

print ' '.join(labels)
#print ' '.join(['Positional', 'Essential'])


print sklearn.metrics.confusion_matrix(ALL_Y_REAL, ALL_Y_PREDICTED, labels=labels)

# labels=['Background', 'Motivation', 'Prior', 'Uses',  'Extends', 'CompareOrContrast', 'Future'])
    
