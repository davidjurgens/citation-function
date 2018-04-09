import sys
import re
from joblib import Parallel, delayed
from fuzzywuzzy import fuzz
from collections import OrderedDict
from collections import defaultdict
#from feature_extraction import *
from feature_extraction import *
from numpy import mean
from scipy import spatial
from nltk.corpus import stopwords

from random import shuffle

reload(sys)  
sys.setdefaultencoding('utf8')


def fix_label(citation_type):
    if citation_type == 'Reference':
        citation_type = 'Background'
    #if citation_type.startswith('Contrast') or citation_type.startswith('Compare'):
    #    citation_type = 'CompareAndContrast'
    return citation_type

CITED_ID_TO_PHRASES = defaultdict(set)
with open('../resources/arc-id-ngrams.2.tsv') as f:
    stop = set([x for x in stopwords.words('english')])
    stop.add('using')
    stop.add('training')
    stop.add('parser')
    stop.add('parser of')
    stop.add('translation')
    stop.add('reranking')
    stop.add('ratio')
    stop.add('similarity')
    stop.add('role')
    stop.add('1999')
    stop.add('entropy')
    stop.add('1999 ;')
    for line in f:
        cols = line.strip().split('\t')
        if len(cols) != 4 or cols[2] != 'pre':
            continue
        tf_idf = float(cols[3])
        
        
        if tf_idf >= 150:
            if cols[1].lower() in stop or (re.search(r'[a-z]', cols[1].lower()) is None):
                # print 'rejecting phrase: ', cols[1]
                continue

            CITED_ID_TO_PHRASES[cols[0]].add(cols[1])            

ARC_ID_TO_TOPIC_VEC = {}
with open('../resources/arc-id-to-topics.2.tsv') as f:
    for line in f:
        cols = line.strip().split('\t')
        arc_id = cols[0]
        vec = [float(x) for x in cols[1].split(' ')]
        ARC_ID_TO_TOPIC_VEC[arc_id] = vec


ARC_ID_TO_CITED_IDs = defaultdict(set)
def load_network():
    with open('../working-files/arc-paper-ids.2.tsv') as f:
        for line in f:
            cols = line[:-1].split('\t')
            ARC_ID_TO_CITED_IDs[cols[0]].add(cols[1])



data_dir = "../data/annotated-json-data/"
teufel_data_dir = "../data/teufel-json/"
mc_data_dir = "../data/meaningful-citation-json-data/"

def main():

    load_network()


    ftrdir = sys.argv[1]

    paper_to_feature_dicts = {}
    
    files_to_process = []
    
    for fname in os.listdir(data_dir):
        
        #if not fname.startswith('D11-1138'):
        #    continue

        #if not fname.startswith('Q13-1020'):
        #    continue

        #if not fname.startswith('J06-3002-parscit'):
        #    continue

        if fname.endswith(".json"):
            files_to_process.append(fname)
            #pass

    for fname in os.listdir(teufel_data_dir):
        if fname.endswith(".json"):
            #if '9503017' in fname:
            files_to_process.append(fname)
            #pass

    for fname in os.listdir(mc_data_dir):
        if fname.endswith(".json"):
            #if '9503017' in fname:
            #files_to_process.append(fname)
            pass


    shuffle(files_to_process)
    # files_to_process = files_to_process[:5]
    
    print 'Processing %d papers' % (len(files_to_process))
    
    #processed_files = [get_paper_features(fname) for fname in files_to_process]
    processed_files = Parallel(n_jobs=-1, verbose=100)(delayed(get_paper_features)(fname) for fname in files_to_process)
    
    for paper_id, feature_dict in processed_files:
        paper_to_feature_dicts[paper_id] = feature_dict

    feature_to_index = {}
    features_in_order = []

    if not os.path.exists(ftrdir):
        os.makedirs(ftrdir)

    #print len(FEATURE_FIRING_COUNTS)
    with open('feature-counts.tsv', 'w') as feat_out:
        for (pattern, feat, clazz), count in FEATURE_FIRING_COUNTS.iteritems():
            feat_out.write('%s\t%s\t%s\t%s\n' % (pattern, feat, clazz, count))


    print 'Indexing features'

    feature_to_all_vals = defaultdict(list)
        
    # Make a quick pass through to assign features to indices
    for paper_id, paper_feature_dicts in paper_to_feature_dicts.iteritems():
        for (labels, paper_feature_dict) in paper_feature_dicts:
            for feature, val in paper_feature_dict.iteritems():
                if feature not in feature_to_index:
                    feature_to_index[feature] = len(feature_to_index)
                    features_in_order.append(feature)
                feature_to_all_vals[feature].append(val)

    feature_to_median_val = {}
    for feat, vals in feature_to_all_vals.iteritems():
        vals.sort()
        feature_to_median_val[feat] = vals[len(vals)/2]

    print 'Saw %d features across %d papers' % (len(feature_to_index), len(paper_to_feature_dicts))
    
    for paper_id, paper_feature_dicts in paper_to_feature_dicts.iteritems():
        outfile = open(ftrdir + '/' + paper_id + ".ftr", "w")
        wr = writer(outfile, delimiter="\t")

        for (labels, paper_feature_dict) in paper_feature_dicts:
            vec = []
            for f in features_in_order:
                if f in paper_feature_dict:
                    vec.append(paper_feature_dict[f])
                else:
                    vec.append(0) # feature_to_median_val[f])
            wr.writerow(labels + vec)
        outfile.close()
                    
    of = open(ftrdir + '/feature-names.txt', 'w')
        #of.write('section_index\nsections_remaining\nyear_difference\n')
    for item in features_in_order:
        of.write("%s\n" % item)
    of.close();

    of = open(ftrdir + '/feature-indices.tsv', 'w')
    for index, item in enumerate(features_in_order):
        of.write("%s\t%d\n" % (item, index))
    of.close();



def get_paper_features(json_fname):

    is_teufel = False

    try:
        with open(data_dir + '/' + json_fname) as jf:
            annotated_data = json.loads(jf.read())
    except:
        try:
            with open(teufel_data_dir + '/' + json_fname) as jf:
                annotated_data = json.loads(jf.read())
                is_teufel = True
        except:
            with open(mc_data_dir + '/' + json_fname) as jf:
                annotated_data = json.loads(jf.read())


    # Need this to exclude certain dep-path data
    annotated_data['file_id'] = json_fname[0:8] 

    #print json_fname
    citing_id = annotated_data['paper_id']

    cited_by_citing = ARC_ID_TO_CITED_IDs[citing_id]
        
    feature_dicts = []    

    cited_id_to_vecs = defaultdict(list)
    cited_id_to_strings = defaultdict(list)
    cited_id_to_locations = defaultdict(set)

    for citation_context in annotated_data['citation_contexts']:       


        
        if not 'citation_type' in citation_context:
            continue

        #if 'External' in citation_context['cited_paper_id']:
        #    continue

        # We already have too many of these...
        if is_teufel and citation_context['citation_role'] == 'Background':
            continue
            # pass

        #print '%s -> %s / %s' % (citation_context['citing_string'], \
        #                             citation_context['citation_role'], \
        #                             citation_context['citation_type'])

        paper_features = get_context_features(citation_context, annotated_data)
              
        if len(paper_features) == 0:
            print "Unable to find context and features in %s for %s; skipping..."\
                % (json_fname, citation_context['citing_string'])
            continue

        # Add the topics of this paper with the idea that some topics have
        # different citing patterns
        if citing_id in ARC_ID_TO_TOPIC_VEC:
            for topic, prob in enumerate(ARC_ID_TO_TOPIC_VEC[citing_id]):
                paper_features['paper_topic_' + str(topic)] = prob
                #pass
    
        cited_id_to_strings[citation_context['cited_paper_id']] \
            .append(citation_context['citing_string'])

        cc = citation_context
        cited_id_to_locations[citation_context['cited_paper_id']] \
            .add((cc['section'], cc['subsection'], cc['sentence']))


        #print paper_features
        cited_id_to_vecs[citation_context['cited_paper_id']] \
            .append((citation_context['citation_type'],  \
                         citation_context['citation_role'], \
                         citation_context['citation_id'], \
                         paper_features))
        

    for cited_id, vecs in cited_id_to_vecs.iteritems():

        features = set()
        global_features = OrderedDict()
        cites_per_section = Counter()

        for t, func, cite_id, vec in vecs:
            # Count how many times this paper is cited in the section
            for feat in vec:
                if feat.startswith('In'):
                    cites_per_section[feat[2:]] += 1
        
        # Count how many citations were in common
        cited_by_cited = ARC_ID_TO_CITED_IDs[cited_id]
        global_features['num_citations_in_common'] = len(cited_by_citing & cited_by_cited)

        # Add a feature for how many times this citation occurs
        global_features['NUM_OCCURRENCES'] = len(vecs)

        # Add features for how many times it was cited per section
        for sec, count in cites_per_section.iteritems():
            global_features['Cites_in_' + sec] = count

        # Add feature for total number of refs
        global_features['total_refs'] = 1.0 / len(cited_id_to_vecs)

        # See how similar the two are
        if cited_id in ARC_ID_TO_TOPIC_VEC \
                and citing_id in ARC_ID_TO_TOPIC_VEC:
            global_features['topic_sim'] = \
                1 - spatial.distance.cosine(ARC_ID_TO_TOPIC_VEC[citing_id], \
                                                ARC_ID_TO_TOPIC_VEC[cited_id])
        else:
            global_features['no_topic_sim'] = 1

        sents_to_exclude = cited_id_to_locations[cited_id]

        total_ind_cites, ind_cites_per_sec \
            = count_indirect_citations(cited_id_to_strings[cited_id], \
                                           CITED_ID_TO_PHRASES[cited_id], \
                                           sents_to_exclude, annotated_data)

        # Add a feature for how many times this indirect citation occurs
        global_features['Num_indirect_cites'] = total_ind_cites

        # Add features for how many times it was indirectly cited per section
        for sec, count in ind_cites_per_sec.iteritems():
            global_features['Indirect_cites_in_' + sec] = count

        #print 'Saw %s indirectly cited %d times, breadown: %s' % (cited_id, total_ind_cites, str(ind_cites_per_sec))

        for cite_type, cite_func, cite_id, vec in vecs:
            vec.update(global_features)
            feature_dicts.append(([cited_id, cite_func, cite_id, cite_type], vec))

    print 'finished', json_fname

    return (annotated_data['paper_id'], feature_dicts)


if __name__ == "__main__":
    main()
