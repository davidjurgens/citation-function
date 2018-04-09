import sys
import re
from joblib import Parallel, delayed
from fuzzywuzzy import fuzz
from collections import OrderedDict
from collections import defaultdict
from numpy import mean
from scipy import spatial
from nltk.corpus import stopwords
import os
from random import shuffle
import fnmatch
reload(sys)  
sys.setdefaultencoding('utf8')
import json        

data_dir = "../data/annotated-json-data/"
teufel_data_dir = "../data/teufel-json/"
arc_dir = '../data/arc-json/'

def main():


    outdir = sys.argv[1]

    citance_contexts_file = outdir + '/citance-contexts.txt'
    extended_citance_contexts_file = outdir + '/extended-citance-contexts.txt'
    context_record_file = outdir + '/context-ids.tsv'


    files_to_process = []   
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            #files_to_process.append(fname)
            files_to_process.append(os.path.join(data_dir, fname))

    for fname in os.listdir(teufel_data_dir):
        if fname.endswith(".json"):
            files_to_process.append(os.path.join(teufel_data_dir, fname))

    for root, dirnames, filenames in os.walk(arc_dir):
        for filename in fnmatch.filter(filenames, '*.json'):
            files_to_process.append(os.path.join(root, filename))
    
    print 'Processing %d papers' % (len(files_to_process))
    
    citance_contexts_outf = open(citance_contexts_file, 'w')
    extended_citance_contexts_outf = open(extended_citance_contexts_file, 'w')

    with open(context_record_file, 'w') as context_record_outf:
        for fi, fname in enumerate(files_to_process):
            get_paper_features(fname, citance_contexts_outf, \
                                   extended_citance_contexts_outf, context_record_outf)
            if (fi + 1) % 50 == 0:
                print 'Processed %d/%d files' % (fi + 1, len(files_to_process))

    #processed_files = Parallel(n_jobs=80, verbose=5)(delayed(get_paper_features)(fname) for fname in files_to_process)
    citance_contexts_outf.close()
    extended_citance_contexts_outf.close()


def get_paper_features(json_fname, ccf, eccf, crf):

    with open(json_fname) as jf:
        annotated_data = json.loads(jf.read())


    for i, citation_context in enumerate(annotated_data['citation_contexts']):
       
        citance = get_citance(citation_context, annotated_data)[0]
        sent = citance['text']
        subsection = get_subsection(citation_context, annotated_data)
        section = annotated_data['sections'][citation_context['section']]
        sent_index = citation_context['sentence']
        context_id = citation_context['citation_id']

        citance, extended_citance = get_topic_contexts(subsection['sentences'], sent_index)

        ccf.write(context_id + " " + json_fname +':' + str(i) + " " + citance.lower() + "\n")
        eccf.write(context_id + " " + json_fname +':' + str(i) + " " + extended_citance.lower() + "\n")
        crf.write(json_fname +'\t' + str(i) + '\t' + context_id + '\n')


def get_citance(citation_context, parsed_doc):
    return parsed_doc['sections'][citation_context['section']]\
        ['subsections'][citation_context['subsection']]\
        ['sentences'][citation_context['sentence']],

def get_subsection(citation_context, parsed_doc):
    return parsed_doc['sections'][citation_context['section']]\
        ['subsections'][citation_context['subsection']]



def get_topic_contexts(sentences, sent_index):
    lemmas = []
    for t in sentences[sent_index]['tokens']:
        lemmas.append(t['lemma'])
    citance = ' '.join(lemmas)

    # previously used -1/+3, but now we just use the whole subsection
    lemmas = []
    #for s in sentences[sent_index-1:sent_index+4]:
    for s in sentences:
        for t in s['tokens']:
            lemmas.append(t['lemma'])
    extended_citance = ' '.join(lemmas)

    return citance, extended_citance


if __name__ == "__main__":
    main()
