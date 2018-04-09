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
from global_functions_march16 import *
from joblib import Parallel, delayed
import ftfy
from unidecode import unidecode
import io

import fnmatch

reload(sys)  
sys.setdefaultencoding('utf8')


def fix_label(citation_type):
    if citation_type == 'Reference':
        citation_type = 'Background'
    #if citation_type.startswith('Contrast') or citation_type.startswith('Compare'):
    #    citation_type = 'CompareAndContrast'
    return citation_type

CITATION_TO_ID = {}
with open("../working-files/arc-citation-network.2.csv","rb") as f:
    rdr = unicodecsv.reader(f, encoding='utf-8', delimiter='\t')
    for entry in rdr:
        paper_id = entry[0]
        year = entry[1]
        title = entry[2]
        authors = tuple(entry[3].split(", "))
        
        CITATION_TO_ID[(year, title, authors)] = paper_id
        paper_data[title] = {}
        paper_data[title]["year"] = year
        paper_data[title]["paper_id"] = paper_id
        paper_data[title]["authors"] = authors

        clean_authors = []
        for name in authors:
            name = unidecode(name)
            name = name.lower().replace("'", '').replace('~','')\
                .replace(",",'').replace('"', '').replace('`', '')
            clean_authors.append(name)
        clean_authors = tuple(clean_authors)
        paper_data[title]["clean_authors"] = clean_authors        
        if clean_authors != authors:
            CITATION_TO_ID[(year, title, clean_authors)] = paper_id

        if title[-4:] == 'quot':
            title = title[0:-4]
        if title[-3:] == ' in':
            title = title[0:-3]
        if title[-5:] == ' corr':
            title = title[0:-5]

        title = re.sub(r'[^a-zA-Z0-9]+', ' ', title)
        title = re.sub(r'[ ]+', ' ', title)
        title = re.sub(r'^[0-9\-]+[a-f]?', '', title)

        clean_title = unidecode(title).replace("'", '').replace('~','')\
            .replace(",",'').replace('"', '').replace('`', '').replace(' ', '')
        CITATION_TO_ID[(year, clean_title, clean_authors)] = paper_id


    #print 'Loaded citation info for %d years' % (len(YEAR_TO_PAPER_DATA))

def get_paper_id(title, year, authors):
    global CITATION_TO_ID
    
    key = (year, title, tuple(authors))

    if key in CITATION_TO_ID:
        return CITATION_TO_ID[key]

    clean_authors = []
    for name in authors:
        name = unidecode(name)
        name = name.lower().replace("'", '').replace('~','')\
            .replace(",",'').replace('"', '').replace('`', '')
        clean_authors.append(name)
        key = (year, title, tuple(clean_authors))
    if key in CITATION_TO_ID:
        return CITATION_TO_ID[key]

    if title[-4:] == 'quot':
        title = title[0:-4]
    if title[-3:] == ' in':
        title = title[0:-3]
    if title[-5:] == ' corr':
        title = title[0:-5]

    title = re.sub(r'[^a-zA-Z0-9]+', ' ', title)
    title = re.sub(r'[ ]+', ' ', title)
    title = re.sub(r'^[0-9\-]+[a-f]?', '', title)

    clean_title = unidecode(title).replace("'", '').replace('~','')\
            .replace(",",'').replace('"', '').replace('`', '').replace(' ', '')
    key = (year, clean_title, tuple(clean_authors))
    if key in CITATION_TO_ID:
        return CITATION_TO_ID[key]

    else:
        print 'Nothing for ', (str(key))
        return ''    

PATH = '../data/arc-parsit-dir/'

def main():


    if len(sys.argv) < 2:
        print 'Usage: convert_ARC_to_features.py out-dir/ [input-dir]'
        return

    output_dir = sys.argv[1]
    input_dir = PATH
    if len(sys.argv) > 2:
        input_dir = sys.argv[2]
    
    num_processed = 0

    #files = [f for f in glob(input_dir + '*.xml')]
    files = []
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, '*.xml'):
            files.append(os.path.join(root, filename))

    print 'Saw %d papers to process' % (len(files))

    #[reformat_paper(fname, output_dir) for fname in files]
    Parallel(n_jobs=64)(delayed(reformat_paper)(fname, output_dir) for fname in files)



def reformat_paper(fname, output_dir):
    

    paper_id = basename(fname)[0:8]
    #print paper_id

    # These are non-standard papers in the ARC (invited talks; tutorials; etc.)
    # which can be skipped
    if not paper_id[1:3].isdigit():
        return

    print fname
    
    try:
        xmldoc = minidom.parse(fname)
    except ExpatError:
        #print 'Bad file?: %s' % (f)
        return


    outfilename = output_dir + '/'  + paper_id + ".json"

    if os.path.isfile(outfilename) and not os.path.getsize(outfilename) == 0:
        #return
        pass

    paper_authors = get_paper_authors(xmldoc)
    parscit_context_to_paper_data = get_context_paper_data(xmldoc)
        
    # Do the text-munging to process the xml
    #text_sents = getPaperSentences_2(xmldoc)
    parsed_doc = parse_paper(xmldoc)

    #entire_doc_text = ' '.join(text_sents)
       
    # Figure out what year this paper was published
    current_paper_year = int(paper_id[1:3])
    if current_paper_year < 50:
        current_paper_year += 2000
    else:
        current_paper_year += 1900

    parsed_doc['year'] = current_paper_year
    parsed_doc['paper_id'] = paper_id

    citation_contexts = []
    parsed_doc['citation_contexts'] = citation_contexts

    itemlist = xmldoc.getElementsByTagName('citation') 
    for s in itemlist:
        try:
            title_elem = s.getElementsByTagName("title")
            if title_elem is None or len(title_elem) == 0 or title_elem[0].firstChild is None:
                continue
            title = cleanTitle(title_elem[0].firstChild.data)
            date_elem = s.getElementsByTagName("date")
            # Need a year
            if date_elem is None or len(date_elem) == 0 or date_elem[0].firstChild is None:
                continue
            year = date_elem[0].firstChild.data
            authors = []
            for elem in s.getElementsByTagName("author"):
                authors.append(elem.firstChild.data)

            raw_string = s.getElementsByTagName("rawString")[0].firstChild.data
                    
            cited_paper_id = get_paper_id(title, year, authors)
            if cited_paper_id == "":
                continue
                                     
        except IndexError, KeyError:
            continue
            

        # Once we've made it through this, we know the ID for the paper, so find
        # its contexts and generate feature vectors for each

        contexts = s.getElementsByTagName("context")
        for context in contexts:
            citing_string = reformat_cite(context.getAttribute("citStr")).encode('utf-8')
            cite_context = context.firstChild.data
            
            cited_authors = authors
            is_self_cit = is_self_cite(paper_authors, cited_authors)


            citing_string_fixed = citing_string

            #print 'Looking for %s in \n%s\n' % (citing_string, cite_context)


            (section, subsection, sentence) = \
                find_citation(cite_context, parsed_doc, citing_string)

            if section < 0 and 'et al ' in citing_string:
                citing_string_fixed = citing_string.replace('et al ', 'et al. ')
                (section, subsection, sentence) = \
                    find_citation(cite_context, parsed_doc, citing_string_fixed)

            if section < 0 and \
                    ('et al. 2' in citing_string or 'et al. 1' in citing_string):
                citing_string_fixed = citing_string.replace('et al. ', 'et al. , ')
                (section, subsection, sentence) = \
                    find_citation(cite_context, parsed_doc, citing_string_fixed)

            if section < 0 and \
                    ('et al 2' in citing_string or 'et al 1' in citing_string):
                citing_string_fixed = citing_string.replace('et al ', 'et al. , ')
                (section, subsection, sentence) = \
                    find_citation(cite_context, parsed_doc, citing_string_fixed)

            # "Dagan and Engelson , 1995" -> "Dagan and Engelson ,1995"
            if section < 0 and ' , ' in citing_string:
                citing_string_fixed = citing_string.replace(' , ', ' ,')
                (section, subsection, sentence) = \
                    find_citation(cite_context, parsed_doc, citing_string_fixed)

            if section < 0:
                # See if we have "Jonson 2000"
                citing_string_fixed = re.sub( r'([a-z]) ([\d]+)', r'\1 , \2', citing_string)
                (section, subsection, sentence) = \
                    find_citation(cite_context, parsed_doc, citing_string_fixed)
                    
            if section < 0:
                print "Unable to find context and features for %s; skipping..." % (citing_string)
                continue

            if citing_string != citing_string_fixed:
                # print 'Fixed: "%s" -> "%s"' % (citing_string, citing_string_fixed)
                citing_string = citing_string_fixed

            #print 'found "%s" in Sec %d, Sub %d, Sent: %d' % (citing_string, section, subsection, sentence)

            citation_context = {
                'citing_string': citing_string,
                'is_self_cite':  is_self_cit,
                'cited_paper_id':  cited_paper_id,
                'info': { 'authors': authors, 'title': title, 'year': year },
                'section': section,
                'subsection': subsection,
                'raw_string': raw_string,
                'sentence': sentence,
                'cite_context': cite_context,
                'citation_id': paper_id + "_" + str(len(citation_contexts))
                }
            citation_contexts.append(citation_context)            
        
    with open(outfilename, 'w') as outf:
        outf.write(json.dumps(parsed_doc))
        outf.write('\n')



# Given the context around the citation (found by ParsCit), determines where
# exactly this citation occurs
def find_citation(parscit_context, parsed_doc, citing_string):


    
    orig_citing_string = citing_string

    context_sentences = to_plain_sentences(parscit_context)
    # Figure out which sentence in the context has the citing string

    # Citation is in the middle
    total_len = 0
    tmp_len = 0
    for cs in context_sentences:
        total_len += len(cs) 
    citing_sent = None
    
    mid_sent_index = 0
    for csi, cs in enumerate(context_sentences):
        tmp_len += len(cs);
        if tmp_len > total_len / 2.0:
            citing_sent = cs
            mid_sent_index = csi
            #print 'Middle sent with %s  should be "%s"' % (citing_string, cs)
            break

    closest_to_mid = 1000
    if citing_sent == None or citing_string not in citing_sent:
        for csi, cs in enumerate(context_sentences):
            if citing_string in cs:
                #if citing_sent is not None:
                #    print 'Ugh; DUPES'
                diff = abs(mid_sent_index - csi)
                if diff < closest_to_mid:
                    closest_to_mid = diff
                    citing_sent = cs

    #print '\nTESTING %s\n%s' % (citing_string, citing_sent)
    if citing_sent == None:
        print 'Could not find "%s" in "%s"' % (citing_string, context_sentences)
        return (-1, -1, -1)

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
                # sents = subsection['sentences'] 
                #context = ' '.join(s['text'] for s in sents[min(0, sent_i-2):max(len(sents), sent_i+3)])

                #print 'best_sent: ', citing_sent
                #print 'sent[text]: ', sent['text']
        
                s = SequenceMatcher(None, citing_sent, sent['text'])
                quick_ratio = s.real_quick_ratio()
                if quick_ratio > best_match:
                    ratio = s.quick_ratio()
                    if best_match < ratio:
                        best_match = ratio
                        best_sec = sec_i
                        best_subsec = subsec_i
                        best_sent = sent_i
                        #print '%s :: New best with %f at %d, %d, %d\n%s' \
                        #    % (citing_string, best_match, sec_i, subsec_i, sent_i, '')

                        
    if False and best_sec >= 0:
        print 'Saw %s in "%s" within context "%s"' \
            % (citing_string,
               parsed_doc['sections'][best_sec]['subsections'][best_subsec]['sentences'][best_sent]['text'],
               parscit_context)
                        
    return (best_sec, best_subsec, best_sent)


def get_valid_elements(root, valid_elements):
    if root.childNodes:
        for node in root.childNodes:
           if node.nodeType == node.ELEMENT_NODE:
               if node.tagName in ['bodyText', 'listItem', 'footnote'] or 'Header' in node.tagName:
                   valid_elements.append(node)
               valid_elements = get_valid_elements(node, valid_elements)
    return valid_elements

def parse_paper(xmldoc):

    text_sents = []
    sent_index_to_section_id = []
    section_id_to_label = {}

    valid_elements = get_valid_elements(xmldoc.documentElement, [])
    sectionPath = []

    cur_section_header = "-1 debug"
    cur_section_index = 0

    # In case some text shows up before a header (due to ParsCit issues)
    section_id_to_label[0] = 'Unknown'

    cur_subsection = { 'sentences': [ ], title: '' }
    cur_section = { 'subsections': [ cur_subsection ], 'num': 0, title: ''}
    sections = [ cur_section ]
    parsed_doc = { 'sections': sections }
    cur_sent_index = 0

    for elem in valid_elements:

        if elem.tagName =="sectionHeader" \
                and float(elem.attributes['confidence'].value) >= 0.8:            
               
            cur_section_header = elem.firstChild.data.strip()
            #print '%s -> %s' % (elem.tagName.strip(), elem.firstChild.data.strip())
            
            cur_section_index += 1
            section_id_to_label[cur_section_index] = cur_section_header

            cur_subsection = { 'sentences': [], title: '' }
            cur_section = { 'subsections': [ cur_subsection ], 'num': cur_section_index, \
                                'title': cur_section_header }
            parsed_doc['sections'].append(cur_section)

        # If we see a subsection, change the section index so we can block
        # sentences by subsection but don't change the header, since we need
        # that as a higher-level feature
        elif elem.tagName =="subsectionHeader" \
                and float(elem.attributes['confidence'].value) >= 0.8:

            cur_section_index += 1
            section_id_to_label[cur_section_index] = cur_section_header
            cur_subsection = { 'sentences': [ ], 'num': 0, \
                                   'title': elem.firstChild.data.strip() }
            cur_section['subsections'].append(cur_subsection)
            

        else:
            text = elem.firstChild.data.strip()
            text = " ".join(text.splitlines())
            text = text.replace("- ", "")
            text = normalize_string(text)        
            #text = text.replace("cf.", "cfXXX")
            
            try:
                cur_subsec_sents = json.loads(corenlp.annotate(text.encode('utf-8'), \
                     properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }).encode('utf-8'), strict=False)
            except ValueError:
                continue

            for sent in cur_subsec_sents['sentences']:
                processed_sent = process_sent(sent)
                cur_subsection['sentences'].append(processed_sent)

    return parsed_doc

def to_plain_sentences(text):

    try:
        sents = json.loads(corenlp.annotate(text.encode('utf-8'), \
           properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }).encode('utf-8'), strict=False)
        plain_sents = []
        for sent in sents['sentences']:
            processed_sent = process_sent(sent)
            plain_sents.append(to_str(processed_sent['tokens']))
        return plain_sents
    except ValueError:
        return []


def process_sent(sentence):
    tokens = sentence['tokens']

    for token in tokens:
        # Fix the issues with possessives, which get assigned to their noun form
        if token['word'] == 'our':
            token['lemma'] = 'our';
        elif token['word'] == 'my':
            token['lemma'] = 'my';
        elif token['word'] == 'their':
            token['lemma'] = 'their';
        elif token['word'] == 'his':
            token['lemma'] = 'his';
        elif token['word'] == 'her':
            token['lemma'] = 'her';
        # Also fix issues with parens...
        elif token['word'] == '-LRB-':
            token['word'] = '('
        elif token['word'] == '-RRB-':
            token['word'] = ')'
        elif token['word'] == '-LSB-':
            token['word'] = '['
        elif token['word'] == '-RSB-':
            token['word'] = ']'



    # Process the dependencies to find the arguments
    deps = sentence['basic-dependencies']
    for dep in deps:
        #print '%s :: %s -> %s' % (dep['dep'], tokens[dep['governor']-1]['word'], tokens[dep['dependent']-1]['word'])

        if dep['dep'] == 'nsubj':
            tokens[dep['dependent']-1]['ArgType'] = 'subj'
            tokens[dep['governor']-1]['ArgType']  = 'verb'
        elif dep['dep'] == 'dobj':
            tokens[dep['dependent']-1]['ArgType'] = 'dobj'            
        else:
            tokens[dep['dependent']-1]['ArgType'] = None

    # Find the root and see if it has a Modal
    root_id = deps[0]['dependent']

    # Mark the verb tense
    if tokens[root_id-1]['pos'][0] == 'V':
        tokens[root_id-1]['is_root'] = True

    if tokens[root_id-1]['pos'][0] == 'V':
        # Use the POS tag tenses
        tokens[root_id-1]['tense'] = tokens[root_id-1]['pos'][-1]

    for dep in deps:
        if dep['dep'] == 'aux' and dep['governor'] == root_id:
            tokens[root_id-1]['has_aux'] = True

        if dep['dep'] == 'auxpass' and dep['governor'] == root_id:
            tokens[root_id-1]['is_pass'] = True

    segment(tokens, deps)
    text = ' '.join(t['word'] for t in tokens)

    return { 'tokens': tokens, 'text': text }


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
