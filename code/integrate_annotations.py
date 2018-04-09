from global_functions_march16 import *
import sys
import re
from joblib import Parallel, delayed
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from os.path import basename

reload(sys)  
sys.setdefaultencoding('utf8')


def fix_label(citation_type):
    if citation_type == 'Reference':
        citation_type = 'Background'
    #if citation_type.startswith('Contrast') or citation_type.startswith('Compare'):
    #    citation_type = 'CompareAndContrast'
    return citation_type

YEAR_TO_PAPER_DATA = {}
with open("../working-files/arc-citation-network.2.csv","rb") as f:
    rdr = unicodecsv.reader(f, encoding='ISO-8859-1',delimiter='\t')
    for entry in rdr:
        paper_id = entry[0]
        year = entry[1]
        title = entry[2]
        authors = entry[3].split(", ")
        
        if year in YEAR_TO_PAPER_DATA:
            paper_data = YEAR_TO_PAPER_DATA[year]
        else:
            paper_data = {}
            YEAR_TO_PAPER_DATA[year] = paper_data    
            
        paper_data[title] = {}
        paper_data[title]["year"] = year
        paper_data[title]["paper_id"] = paper_id
        paper_data[title]["authors"] = authors
    print 'Loaded citation info for %d years' % (len(YEAR_TO_PAPER_DATA))

def getPaperId(title, year, authors):
    global EXTERNAL_PAPER_IDS
    if year not in YEAR_TO_PAPER_DATA:
        #print '%s not in %s' % (year, YEAR_TO_PAPER_DATA.keys()) 
        return ''
    else:
        paper_data = YEAR_TO_PAPER_DATA[year]
    
    # Everything should be in this 
    if title in paper_data:
        d = paper_data[title]
        return d["paper_id"]
    else:
        #print 'Not entry for ' + title
        return ''

def store_citation_labels(fname):
    labels = {}
    perfunctory_ids = set()
    id_to_offset = {}
    id_to_name = {}
    id_to_citing_string = {}
    fr = open(fname)
    for l in fr:
        ls = l.strip().split("\t")
        if ls[0].startswith("T"):
            # CiteType start_offset end_offset
            mid_cols = ls[1].split(" ")
            citation_type = fix_label(mid_cols[0])

            if citation_type == 'Unknown':
                continue

            labels[ls[0]] = citation_type
            # print ls[0], citation_type
            id_to_offset[ls[0]] = (int(mid_cols[1]), int(mid_cols[2]))
            id_to_citing_string[ls[0]] = ls[2]
        elif ls[0].startswith("A"):
            perfunctory_ids.add(ls[1].split(" ")[1])

    return labels, perfunctory_ids, id_to_offset, id_to_citing_string


JSON_DIR = "../data/arc-json/"
ANNOTATION_DIR = "../data/adjudicated-and-supplemental/"


def main():

    output_dir = sys.argv[1]    
    files_to_process = []
    
    for fname in os.listdir(ANNOTATION_DIR):
        
        #if not fname.startswith('N01-1009'):
        #    continue

        #if not fname.startswith('Q13-1020'):
        #    continue

        #if not fname.startswith('J06-3002-parscit'):
        #    continue

        if fname.endswith(".ann"):
            files_to_process.append(fname)

    print 'Processing %d papers' % (len(files_to_process))

    processed_files = [integrate(anno_fname, output_dir) for anno_fname in files_to_process]
    #processed_files = Parallel(n_jobs=64)(delayed(get_paper_features)(fname, xmldir, data_dir) for fname in files_to_process)


total_num_integrated = 0
total_num_seen = 0

def integrate(anno_fname, output_dir):

    global total_num_integrated
    global total_num_seen

    
    id_to_label, id_is_perfunctory, id_to_offset, id_to_citing_string = \
        store_citation_labels(ANNOTATION_DIR + anno_fname)

    # print id_to_label

    paper_id = basename(anno_fname)[0:8]

    txt_file = anno_fname.replace('.ann', '.txt')
    json_file = JSON_DIR + paper_id + '.json'

    if not os.path.isfile(json_file) or os.path.getsize(json_file) == 0:
        print 'No json file: %s ; skipping' % (json_file)
        return


    print paper_id

    with open(json_file) as jf:
        parsed_doc = json.loads(jf.read())

    with open(ANNOTATION_DIR + "/" + txt_file) as fh:
        annotated_text = fh.read()
    
    num_integrated = 0
    total_num_seen += len(id_to_label)

    for anno_id, label in id_to_label.iteritems():
        # Figure out where is the ParsCit context in the annotated text
        (start, end) = id_to_offset[anno_id]
        # Get the the specific citation that was annotated
        citing_string = reformat_cite(id_to_citing_string[anno_id])
        # Get the citation type
        citation_role = id_to_label[anno_id]

        is_positional = anno_id in id_is_perfunctory
                
        # print "\n%s ==> %s (%s)" % (citing_string, citation_role, str(is_positional))

        line_start = annotated_text.rfind('\n', 0, start)
        line_end = annotated_text.find('\n', end, len(annotated_text))
        
        # print '%s :: (%d, %d) -> (%d, %d)' % (citing_string, start, end, line_start, line_end)
        # This happens for the first annotation
        if line_start < 0:
            line_start = 0
        
        parscit_context = annotated_text[line_start:line_end]

        best_match = -1
        best_context = None

        for citation_context in parsed_doc['citation_contexts']:

            s = SequenceMatcher(None, citation_context['citing_string'], citing_string)
            if s.ratio() < 0.95:
                #print '%s != %s :: (%f)' % (citation_context['citing_string'], citing_string, s.ratio())
                continue

            #print 'YAY::  %s == %s :: (%f)' % (citation_context['citing_string'], citing_string, s.ratio())
            cite_context = citation_context['cite_context']

            s = SequenceMatcher(None, cite_context, parscit_context)
            quick_ratio = s.real_quick_ratio()
            if quick_ratio > best_match:
                ratio = s.quick_ratio()
                if best_match < ratio:
                    best_match = ratio
                    best_context = citation_context

        if best_match < 0.8:
            #print 'unable to find %s (%f) in context "%s"' % (citing_string, best_match, parscit_context)
            #print 'unable to find %s; best %s' % (citing_string, str(best_match))
            continue

        #print 'matched with score %f:\nPARCIT: %s\nORIG: %s\n' % (best_match, parscit_context, best_context['cite_context'])

        num_integrated += 1
        best_context['citation_role'] = citation_role
        if is_positional:
            best_context['citation_type'] = 'Positional'
        else:
            best_context['citation_type'] = 'Essential'

    total_num_integrated += num_integrated
    print 'Integrated %d / %d annotations (%d / %d total)' % \
        (num_integrated, len(id_to_label), total_num_integrated, total_num_seen)

    outfilename = output_dir + '/'  + paper_id + ".json"
    with open(outfilename, 'w') as outf:
        outf.write(json.dumps(parsed_doc))
        outf.write('\n')

            


# Given the context around the citation (found by ParsCit), determines where
# exactly this citation occurs
def find_citation(parscit_context, parsed_doc, citing_string):
    
    orig_citing_string = citing_string


    best_match = -1
    best_sec = -1
    best_subsec = -1
    best_sent = -1

    for sec_i, section in enumerate(parsed_doc['sections']):
        for subsec_i, subsection in enumerate(section['subsections']):
            for sent_i, sent in enumerate(subsection['sentences']):

                # Find only sentences where the citation occurs
                if not citing_string in sent['text']:
                    continue

                # Grab +/-2 sentences for more context
                sents = subsection['sentences'] 
                context = ' '.join(s['text'] for s in sents[min(0, sent_i-2):max(len(sents), sent_i+3)])
        
                s = SequenceMatcher(None, context, parscit_context)
                quick_ratio = s.real_quick_ratio()
                if quick_ratio > best_match:
                    ratio = s.quick_ratio()
                    if best_match < ratio:
                        best_match = ratio
                        best_sec = sec_i
                        best_subsec = subsec_i
                        best_sent = sent_i

                        
    if False and best_sec >= 0:
        print 'Saw %s in "%s" within context "%s"' \
            % (citing_string,
               parsed_doc['sections'][best_sec]['subsections'][best_subsec]['sentences'][best_sent]['text'],
               parscit_context)
                        
    return (best_sec, best_subsec, best_sent)


def reformat_cite(citing_string):
    result = json.loads(corenlp.annotate(citing_string.encode('utf-8'), properties={'annotators': 'tokenize,ssplit' }).encode('utf-8'), strict=False)
    tokens = []
    # iterate over all sentences -even though there should only be one- just in
    # case CoreNLP messes up somehow
    for sent in result['sentences']:
        for token in sent['tokens']:
            if token['word'] == '-LRB-':
                token['word'] = '('
            elif token['word'] == '-RRB-':
                token['word'] = ')'
            elif token['word'] == '-LSB-':
                token['word'] = '['
            elif token['word'] == '-RSB-':
                token['word'] = ']'

            tokens.append(token['word'])
    return ' '.join(tokens)


if __name__ == "__main__":
    main()
