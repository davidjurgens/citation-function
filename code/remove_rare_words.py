import sys
import re
from joblib import Parallel, delayed
from fuzzywuzzy import fuzz
from collections import OrderedDict
from collections import Counter
from numpy import mean
from scipy import spatial
from nltk.corpus import stopwords
import os
from random import shuffle
import fnmatch
reload(sys)  
sys.setdefaultencoding('utf8')
import json        


MIN_FREQ=2

def main():

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    wc = Counter()
    
    with open(input_file) as f:
        for line in f:
            for w in line.split()[2:]:
                wc[w] += 1
                


    auto_stopwords = set([x[0] for x in wc.most_common(200)])
    for w, c in wc.iteritems():
        if c < MIN_FREQ:
            auto_stopwords.add(w)

    print 'Trimmed vocab from %d to %d' % (len(wc), len(wc) - len(auto_stopwords))

    with open(input_file) as f, open(output_file, 'w') as outf:
        for line in f:
            cols = line.split()
            outf.write(cols[0] + ' ' + cols[1] + ' ')
            words = [w for w in cols[2:] if w not in auto_stopwords]
            if len(words) == 0:
                # print 'bad!!'
                words.append('DUMMY')
            outf.write(' '.join(words))
            outf.write('\n')

    
                
if __name__ == '__main__':
    main()
