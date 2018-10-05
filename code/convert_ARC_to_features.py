from xml.dom import minidom
from glob import glob
from os.path import basename
import unicodecsv
import string
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import sys
from xml.parsers.expat import ExpatError
import sys
import re
import fnmatch
from feature_extraction import *
from joblib import Parallel, delayed
from random import shuffle
from numpy import mean
from scipy import spatial
from nltk.corpus import stopwords
import os

reload(sys)  
sys.setdefaultencoding('utf8')

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
                #print 'rejecting phrase: ', cols[1]
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
    with open('../resources/arc-citation-network.tsv ') as f:
        for line in f:
            cols = line[:-1].split('\t')
            ARC_ID_TO_CITED_IDs[cols[0]].add(cols[1])


PATH = '../data/arc-json/'

def main(): 

    if len(sys.argv) < 2:
        print 'Usage: convert_ARC_to_features.py arc-json-dir/ ftr-dir/ out-dir/'
        return

    input_dir = sys.argv[1]
    ftr_dir = sys.argv[2]
    output_dir = sys.argv[3] + '/'

    feature_to_index = {}
    with open(ftr_dir + '/feature-indices.tsv') as f:
        for line in f:
            cols = line.strip().split('\t');
            feature = cols[0]
            index = int(cols[1])
            feature_to_index[feature] = index

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    num_processed = 0

    #files = [f for f in glob(input_dir + '*.xml')]
    files = []
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, '*.json'):
            #if 'W06-3319' in filename:
            files.append(os.path.join(root, filename))
 
    shuffle(files)

    # files = files[:10]

    print 'Saw %d papers to process' % (len(files))

    #[create_feature_file(fname, output_dir) for fname in files]
    Parallel(n_jobs=-1)(delayed(create_feature_file)(fname, feature_to_index, output_dir) for fname in files)




def create_feature_file(f, feature_to_index, output_dir):

    paper_id = basename(f)[0:8]

    outfilename = output_dir + basename(f)[0:8] + ".ftr"

    if os.path.isfile(outfilename) and not os.path.getsize(outfilename) == 0:
        print 'already created %s ; skipping ' % (outfilename)
        return

    if not paper_id[1:3].isdigit():
        print 'unexpected filename %s ; skipping ' % (basename(f))
        return
        


    with open(f) as jf:
        json_data = json.loads(jf.read())
        citation_vecs = get_paper_features(json_data)
        if len(citation_vecs) == 0:
            print 'no citations in %s ; skipping ' % (basename(f))
            return

        outfile = open(outfilename, 'w')
        wr = writer(outfile, delimiter="\t")
      
        print paper_id

        for cited_id, citation_id, feature_dict in citation_vecs:
                    
            vec = [0] * len(feature_to_index)
            for f, val in feature_dict.iteritems():
                if not f in feature_to_index:
                    # print 'Saw new feature not in data: %s' % (f)
                    continue
                vec[feature_to_index[f]] = val
            wr.writerow([paper_id, cited_id, citation_id] + vec)

        outfile.close()


    print 'done with %s, wrote to %s' % (paper_id, outfilename)

def get_paper_features(annotated_data):

    #print json_fname
    citing_id = annotated_data['paper_id']
    cited_by_citing = ARC_ID_TO_CITED_IDs[citing_id]

    annotated_data['file_id'] = None
        
    feature_dicts = []    

    cited_id_to_vecs = defaultdict(list)
    cited_id_to_strings = defaultdict(list)
    cited_id_to_locations = defaultdict(set)

    for citation_context in annotated_data['citation_contexts']:       

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
    
        cited_id_to_strings[citation_context['cited_paper_id']] \
            .append(citation_context['citing_string'])

        cc = citation_context
        cited_id_to_locations[citation_context['cited_paper_id']] \
            .add((cc['section'], cc['subsection'], cc['sentence']))


        #print paper_features
        cited_id_to_vecs[citation_context['cited_paper_id']] \
            .append((citation_context['citation_id'], paper_features))


    for cited_id, vecs in cited_id_to_vecs.iteritems():

        features = set()
        global_features = OrderedDict()
        cites_per_section = Counter()

        for citation_id, vec in vecs:
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

        for citation_id, vec in vecs:
            vec.update(global_features)
            feature_dicts.append((cited_id, citation_id, vec))

    return feature_dicts

def get_paper_features_old(annotated_data):

    #print json_fname
    citing_id = annotated_data['paper_id']
        
    feature_dicts = []    

    cited_id_to_vecs = defaultdict(list)
    cited_id_to_strings = defaultdict(list)
    cited_id_to_locations = defaultdict(set)

    for citation_context in annotated_data['citation_contexts']:       
        
        paper_features = get_context_features(citation_context, annotated_data)
              
        if len(paper_features) == 0:
            #print "Unable to find context and features in %s for %s; skipping..."\
            #    % (json_fname, citation_context['citing_string'])
            continue
    
        cited_id_to_strings[citation_context['cited_paper_id']] \
            .append(citation_context['citing_string'])

        cc = citation_context
        cited_id_to_locations[citation_context['cited_paper_id']] \
            .add((cc['section'], cc['subsection'], cc['sentence']))


        #print paper_features
        cited_id_to_vecs[citation_context['cited_paper_id']] \
            .append(paper_features)


    for cited_id, vecs in cited_id_to_vecs.iteritems():


        features = set()
        for vec in vecs:
            for f in vec:
                features.add(f)

        merged_vec = OrderedDict()
        cites_per_section = Counter()

        if len(vecs) == 1:
            merged_vec = vecs[0]
        else:
            for feat in features:
                vals = []

                # Merge all the dicts
                for vec in vecs:
                    if feat in vec:
                        vals.append(vec[feat])

                if 1 == 1:
                    # Count how many times this paper is cited in the section
                    if feat.startswith('In'):
                        cites_per_section[feat[2:]] = len(vals)

                    # For text-pattern features, use the minimum, indicating how
                    # close the phrase ever was
                    if '_AGENT' in feat or '_FORMULAIC' in feat \
                            or 'DISTANCE_TO' in feat or 'Connector_' in feat:
                        merged_vec[feat] = min([abs(x) for x in vals])

                    # Section Numbers get treated specially so that we mark
                    # which sections the citation occurs in
                    elif 'Section_Num' == feat:
                        for v in vals:
                            merged_vec['Section_Num_' + str(v)] = len(vals)
                    # For topic models, use the maximum, indicating that the
                    # citation appeared in some context that had a lot of this
                    # topic
                    elif '_Topic' in feat:
                        merged_vec[feat] = max(vals)
                    # For features describing the number of citations, relative
                    # position, or length, use the average
                    elif feat.startswith('NUM_') \
                            or '_length' in feat \
                            or 'relative_pos' in feat \
                            or 'osition' in feat:
                        merged_vec[feat] = mean(vals)                        
                    else:
                        #print 'default action for ', feat
                        merged_vec[feat] = max(vals)

        # Add a feature for how many times this citation occurs
        merged_vec['NUM_OCCURRENCES'] = len(vecs)

        # Add features for how many times it was cited per section
        for sec, count in cites_per_section.iteritems():
            merged_vec['Cites_in_' + sec] = count

        # Add feature for total number of refs
        merged_vec['total_refs'] = 1.0 / len(cited_id_to_vecs)

        # See how similar the two are
        if cited_id in ARC_ID_TO_TOPIC_VEC \
                and citing_id in ARC_ID_TO_TOPIC_VEC:
            merged_vec['topic_sim'] = \
                1 - spatial.distance.cosine(ARC_ID_TO_TOPIC_VEC[citing_id], \
                                                ARC_ID_TO_TOPIC_VEC[cited_id])
        else:
            merged_vec['no_topic_sim'] = 1

        sents_to_exclude = cited_id_to_locations[cited_id]

        total_ind_cites, ind_cites_per_sec \
            = count_indirect_citations(cited_id_to_strings[cited_id], \
                                           CITED_ID_TO_PHRASES[cited_id], \
                                           sents_to_exclude, annotated_data)

        # Add a feature for how many times this indirect citation occurs
        merged_vec['Num_indirect_cites'] = total_ind_cites

        # Add features for how many times it was indirectly cited per section
        for sec, count in ind_cites_per_sec.iteritems():
            merged_vec['Indirect_cites_in_' + sec] = count

        #print 'Saw %s indirectly cited %d times, breadown: %s' % (cited_id, total_ind_cites, str(ind_cites_per_sec))

        feature_dicts.append((cited_id, merged_vec))

    print 'finished', citing_id

    return feature_dicts



if __name__ == "__main__":
    main()
