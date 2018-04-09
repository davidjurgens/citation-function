import unicodecsv
import json
import re
import numpy as np
from numpy import histogram
import cPickle
from os import listdir
import math
from collections import defaultdict
import pandas as pd
import sklearn
import random
import csv
import sys
import os
from os.path import basename
import copy
from sklearn.ensemble import RandomForestClassifier
import random
from collections import Counter
import sys
import pickle


csv.field_size_limit(sys.maxsize)

if len(sys.argv) < 4:
    print 'usage: python classify-papers.py classifier.pkl paper.ftr output.tsv [num topics]'
    quit(1)


feature_dir = "../working-files/feature-vectors/"

clf_pickle = sys.argv[1]
feature_file = sys.argv[2]
outfile = sys.argv[3]

num_topics = 50
if len(sys.argv) > 4:
    num_topics = int(sys.argv[4])

if os.path.isfile(outfile) and not os.path.getsize(outfile) == 0:
    print 'Already generated %s ; skipping' % (outfile)
    quit(0)
    
#clf = joblib.load(clf_pickle) 
with open(clf_pickle, 'rb') as f:
    clf = pickle.load(f)
print 'loaded pickle'

feature_to_index = {}
with open(feature_dir + 'feature-indices.tsv') as f:
    for line in f:
        cols = line.strip().split('\t');
        feature = cols[0]
        index = int(cols[1])
        feature_to_index[feature] = index
print 'loaded feature index'

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


fname = basename(feature_file)
arr = fname.split('-')
paper_id = (arr[0] + '-' + arr[1]).replace('.ftr', '')
print paper_id
    
#topic_dir = '/lsfs/jurgens/citation/'
#if not os.path.isfile(topic_dir):
#    topic_dir = '/lfs/local/0/jurgens/citation/'
#if not os.path.isfile(topic_dir):
topic_dir = '../working-files/'



citation_id_to_topic_vecs = {}
with open(topic_dir + '/topics2/%d-topics/citance.doc-topics.txt' % num_topics) as citance_topic_f:
    with open(topic_dir + '/topics2/%d-topics/extended-citance.doc-topics.txt' % num_topics) as extended_topic_f:
        with open('../working-files/context-ids.tsv') as f:
            for line in f:
                cols = line[:-1].split('\t')
                citation_id = cols[2]

                citance_topic_line = citance_topic_f.readline()
                extended_topic_line = extended_topic_f.readline()

                if not citation_id.startswith(paper_id):
                    continue


                citance_topic_vec = np.array(map(float, citance_topic_line[:-1].split('\t')[2:]))
                extended_topic_vec = np.array(map(float, extended_topic_line[:-1].split('\t')[2:]))
                
                citation_id_to_topic_vecs[citation_id] = (citance_topic_vec, extended_topic_vec)

print 'loaded %d topics' % (len(citation_id_to_topic_vecs))

count = 0


with open(outfile, 'w') as ofw:
    with open(feature_file, "r") as f:
        rdr = csv.reader(f, delimiter="\t")
        to_append = []
        for cols in rdr:
            if len(cols) < 10:
                continue
        
            citing_id = cols[0]
            cite_id = cols[1]
            citation_id = cols[2]
        
            feature_vec = np.array(map(float, cols[3:]))

            # TODO: Append the topic vector here
            topic_vecs = citation_id_to_topic_vecs[citation_id]
            
            feature_vec = np.concatenate([feature_vec, topic_vecs[0], topic_vecs[1]])           

            feature_vec = feature_vec.reshape(1, -1)

            section = get_section(feature_vec[0])

            predicted_y = clf.predict(feature_vec)[0]
            ofw.write('\t'.join([citing_id, cite_id, citation_id, section, predicted_y]) + '\n')
        
            count += 1

        print 'finished ', fname

