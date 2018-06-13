import unicodecsv
from xml.dom import minidom
import optparse
import os
import sys
import re
#from urllib2 import Request, build_opener, HTTPCookieProcessor
import urllib
import urllib2
from cookielib import MozillaCookieJar
import subprocess
from bs4 import BeautifulSoup as bs
import shlex
import time 
import nltk
import string 
import numpy
import json
import nltk.data
from csv import writer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from collections import OrderedDict
from collections import Counter
import math 
from difflib import SequenceMatcher
from pycorenlp import StanfordCoreNLP
from fuzzywuzzy import fuzz

reload(sys)  
sys.setdefaultencoding('utf8')



LMTZR = WordNetLemmatizer()    

corenlp = StanfordCoreNLP('http://localhost:8999')

PARSED_DOC_SENTENCE_CACHE = {}
DOC_SENTENCE_CACHE = {}

debug_feat_descriptions = []

def cleanTitle(title):
    exclude = set(string.punctuation)
    title = title.lower().strip()
    title = ''.join(ch for ch in title if ch not in exclude)
    return title

paper_data = {}

def load_patterns(filename, p_dict, label):
    with open(filename) as f:
        class_counts = Counter()
        for line in f:
            if not '@' in line:
                continue
            cols = line.split("\t")
            pattern = cols[0].replace("-lrb-", "(").replace('-rrb-', ')')
            clazz = cols[1]
            if clazz == 'Background':
                continue
            class_counts[clazz] += 1
            p_dict[clazz + '_' + label + '_' + str(class_counts[clazz])] \
                = [pattern.split()]
            #p_dict[clazz + '_' + label].append(pattern.split())

CUSTOM_IN_CITANCE_PATTERNS   = defaultdict(list)
CUSTOM_PRE_CITANCE_PATTERNS  = defaultdict(list)
CUSTOM_POST_CITANCE_PATTERNS = defaultdict(list)

#load_patterns('pattern-type-counts.in-citance.filtered.tsv', \
#                  CUSTOM_IN_CITANCE_PATTERNS, 'in_cite')
#load_patterns('pattern-type-counts.preceding.filtered.tsv', \
#                  CUSTOM_PRE_CITANCE_PATTERNS, 'pre_cite')
#load_patterns('pattern-type-counts.following.filtered.tsv', \
#                  CUSTOM_POST_CITANCE_PATTERNS, 'post_cite')

load_patterns('../resources/patterns/in-sent.filtered.tsv', \
                  CUSTOM_IN_CITANCE_PATTERNS, 'in_cite')
load_patterns('../resources/patterns/before.filtered.tsv', \
                  CUSTOM_PRE_CITANCE_PATTERNS, 'pre_cite')
load_patterns('../resources/patterns/after.filtered.tsv', \
                  CUSTOM_POST_CITANCE_PATTERNS, 'post_cite')



def get_paper_authors(root):
    if root.childNodes:
        for node in root.childNodes:
           if node.nodeType == node.ELEMENT_NODE:
               #print node.tagName
               if node.tagName == 'author':
                   return node.firstChild.data.strip()
               else:
                   authors = get_paper_authors(node)
                   if authors != '':
                       return authors
               
    return ''


def get_context_paper_data(xmldoc):
    itemlist = xmldoc.getElementsByTagName('citation') 
    context_to_paper_data = {}
    
    for s in itemlist:
        paper_data = {}
        authors = []
        paper_data['authors'] = authors
        paper_data['title'] = ''
        paper_data['year'] = -1
        
        try:
            for elem in s.getElementsByTagName("author"):
                authors.append(elem.firstChild.data)
        except BaseException:
            continue

        try:
            for elem in s.getElementsByTagName("title"):
                paper_data['title'] = cleanTitle(elem.firstChild.data)
                break
        except BaseException:
            continue

        try:
            for elem in s.getElementsByTagName("date"):
                #print str(elem.firstChild)
                paper_data['year'] = elem.firstChild.data
                break
        except BaseException:
            continue

        contexts = s.getElementsByTagName("context")
        for context in contexts:
            cite_context = context.firstChild.data
            context_to_paper_data[cite_context] = paper_data
            #print '%s wrote %s' % (str(authors), cite_context)

    return context_to_paper_data

def find_paper_data(parscit_context_to_paper_data, \
                        parscit_context):

    best_fuzzy_overlap = 80
    best_data = {'title': '', 'authors': [], 'year': -2}
    best_context = ''

    for context, paper_data in parscit_context_to_paper_data.iteritems():
        ratio = fuzz.ratio(context, parscit_context)
        if ratio < best_fuzzy_overlap:
            continue

        best_fuzzy_overlap = ratio
        best_data = paper_data
        best_context = context

    #print 'Found match at %d\n\t%s\n\t%s' % (best_fuzzy_overlap, parscit_context, str(best_data))
    return best_data

def is_self_cite(paper_authors, cited_authors):
    # paper_authors is just a string of names
    # cited authors is a list itself
    pauth_names = paper_authors.split()

    for author in cited_authors:
        author = author.split()
        for i in range(0, len(pauth_names)-(len(author)-1)):
            m = True
            for j, name in enumerate(author):
                #print '%s starts with %s? %s' % \
                #    (pauth_names[i+j], name, str(pauth_names[i+j].startswith(name)))
                if not pauth_names[i+j].startswith(name):
                    m = False
                    break
            if m:
                #print 'Self-cite: Found %s in %s' % (str(author), paper_authors)
                return True
    return False                



def normalize_section(section):
    if section == []:
        return ""
    ss = section.replace('\n', ' ').split(" ")
    vals = []
    for sss in ss:
       if not sss.replace(".","").strip().isdigit():
           vals.append(sss)
    return " ".join(vals)

def get_all_sections(xmldoc):
    sections = xmldoc.getElementsByTagName('sectionHeader')
    section_names = []
    for section in sections:
        section_names.append(normalize_section(section.firstChild.data).strip())
    return section_names


def get_section_title_feature(section_name):
    lc = section_name.replace(" ", "").lower()
    features = {} 
    if lc == 'abstract':
        features['InAbstract'] = 1
    elif "intro" in lc:
        features['InIntroduction'] = 1
    elif "related" in lc or "existing" in lc or "previous" in lc or 'background' in lc:
        features['InRelatedWork'] = 1
    elif "motivation" in lc or 'applications' in lc:
        features['InMotivation'] = 1

    elif "method" in lc or "implementation" in lc or 'approach' in lc \
            or 'model' in lc or 'system' in lc or 'algorithm' in lc:
        features['InMethology'] = 1

    elif "evalu" in lc or"experiment" in lc  or 'task' in lc or 'data' in lc \
            or 'setup' in lc or 'corpus' in lc or 'problem' in lc:
        features['InEvaluation'] = 1

    elif "results" in lc or 'analysis' in lc:
        features['InResults'] = 1

    elif "discussion" in lc:
        features['InDiscussion'] = 1
    elif "conclusion" in lc or 'conclud' in lc or 'final remark' in lc \
            or 'future work' in lc or 'future research' in lc:
        features['InConclusion'] = 1
    elif "acknowl" in lc:
        features['InAcknowledgements'] = 1
    elif lc.startswith('reference') or 'bibliography' in lc:
        features['InReferences'] = 1

    return features


CUR_LABEL=None
FEATURE_FIRING_COUNTS=Counter()



#
# The input is one citation in one paper.  If the paper is cited multiple times,
# parcit_context is the context for one citation in one paper.
#
def get_citation_context_and_features(parscit_context, is_self_citation, text_sents, \
                                          sent_index_to_section_id, \
                                          section_id_to_label, entire_doc_text, \
                                          citing_string, doc_label, debug_label=None):    

    global CUR_LABEL
    

    if debug_label is not None:
        CUR_LABEL = debug_label

    if doc_label in PARSED_DOC_SENTENCE_CACHE:
        parsed_text_sents = PARSED_DOC_SENTENCE_CACHE[doc_label]    
    else:
        PARSED_DOC_SENTENCE_CACHE.clear()
        parsed_text_sents = {}
        PARSED_DOC_SENTENCE_CACHE[doc_label] = parsed_text_sents


    # this is the citation itself, e.g., "Foo (2000)"
    citingString = citing_string    

    idx = find_citation(parscit_context, text_sents, \
                            entire_doc_text, citing_string)

    # If we couldn't find it :(
    if idx < 0:
        return {}

    if True:
        #return {'a': 1}
        pass

    # This is the citing sentence
    sent = text_sents[idx]
    _number_of_sentences_in_context = 3


    CITATION = 'CITATION'
    if is_self_citation:
        CITATION = 'SELFCITATION'

    #
    #
    # This section begins the feature extract part 
    #
    #
    

    IS_USED_AS_TEXT = 0
    if '(' in citingString or ')' in citingString:
        IS_USED_AS_TEXT = 1

    

    DOES_SENTENCE_START_WITH_CITATION = int(sent.startswith(citingString))

    # +/-3 sentences before and after, so 7 sentences in total
    CONTEXT = " ".join(text_sents[idx-_number_of_sentences_in_context : idx+_number_of_sentences_in_context + 1])

    # The 3 sentences before or after the citation
    PRE_CONTEXT = " ".join(text_sents[idx-_number_of_sentences_in_context : idx])
    POST_CONTEXT = " ".join(text_sents[idx + 1: idx+_number_of_sentences_in_context + 1])
        
    # The index of the sentence containing the citation
    SENTENCE_INDEX = idx


    # Number of numbers in the citation sentence, in context
    NUM_NUMBERS_IN_CITATION_SENTENCE = -1 #get_number_of_numbers_in_text(sent)
    NUM_NUMBERS_IN_CONTEXT = -1 #get_number_of_numbers_in_text(CONTEXT)

    citePositionInSentence = sent.index(citingString) / float(len(sent))
        
    # Grab the text right before the citation
    ref_index = sent.find(citingString)
    sentence_text_before_citation = sent[:ref_index].strip()
    sentence_text_after_citation = sent[(ref_index + len(citingString)):].strip()
    

    # Do so munging to strip off (a) prior citations if this is in the middle of
    # a citation block, (b) trailing citations, (c) parentheses
    i = sentence_text_before_citation.rfind('(')
    if i >= 0:
        sentence_text_before_citation = sentence_text_before_citation[:i].strip()
        
    i = sentence_text_after_citation.find(')')
    if i >= 0:
        sentence_text_after_citation = sentence_text_after_citation[(i+1):].strip()
       
    if "(" in citingString:
        citingString_bracket = citingString.split("(")[1].split(")")[0]
    else:
        citingString_bracket = citingString

    NUM_CITATIONS_IN_SAME_CITATION = 0

    IS_CAPITALIZED = 0
    IS_CONJUNCTION = 0

    is_cite_in_parens = not ('(' in citing_string and ')' in citing_string)    
    IS_CITE_IN_PARENS = 0
    if is_cite_in_parens:
        IS_CITE_IN_PARENS = 1


    pos_of_cite = sent.find(citing_string)
    full_cite = citing_string
    if is_cite_in_parens:
        cite_start = sent.rfind('(', 0, pos_of_cite)        
        #print 'Searching for ( before %s, found at %d before %d' \
        #    % (citing_string, cite_start, pos_of_cite)
        if cite_start >= 0:
            pos_of_cite = cite_start
            cite_end = sent.find(')', pos_of_cite)
            full_cite = sent[cite_start:cite_end+1]

    preceding_text = sent[0:pos_of_cite].strip()
    anteceding_text = sent[pos_of_cite+len(full_cite):].strip()
    IS_EXAMPLE = 0

    #print 'Given sentence: ' + sent
    #print 'Citation occurs at [%d, %d]' % 
    #print 'PREC: "%s"\nANTE: "%s"' % (preceding_text, anteceding_text)

    if preceding_text.endswith('e.g.,') \
            or preceding_text.endswith('e.g.') \
            or preceding_text.endswith('for example') \
            or preceding_text.endswith('for example,') \
            or preceding_text.endswith('for instance') \
            or preceding_text.endswith('for instance,') \
            or preceding_text.endswith('see') \
            or preceding_text.endswith('cf.') \
            or preceding_text.endswith('cf'):
        #print 'Saw %s in example in sent: %s' % (citing_string, sent);
        IS_EXAMPLE = 1
    elif is_cite_in_parens and (full_cite.startswith("(e.g.") \
                                    or full_cite.startswith("(cf.")):
        #print 'Saw %s in example in sent: %s' % (citing_string, sent);
        IS_EXAMPLE = 1        
    elif anteceding_text.startswith(", for example,") \
            or anteceding_text.startswith(", for instance,"):
        #print 'Saw %s in example in sent: %s' % (citing_string, sent);
        IS_EXAMPLE = 1



    # Look for specific structural features that preceed non-textual citations,
    # e.g., (Foo, 2000), which can indicate the citation is referring to
    # something like an algorithm
    CAMELCASE = 0
    ALL_CAPS = 0
    if is_cite_in_parens:
        preceding_tokens = preceding_text.split()
        if len(preceding_tokens) > 0:
            for pi in range(len(preceding_tokens) - 1, len(preceding_tokens)-4, -1):
                if pi < 0:
                    break            
                prev_word = preceding_tokens[pi]
                if ',' in prev_word or ')' in prev_word or ';' in prev_word:
                    break
                #print '"%s" occurs before %s in sent:: %s' % (prev_word, citing_string, sent)
                if prev_word.isupper():
                    ALL_CAPS = 1
                elif prev_word[0:1].isupper() and not prev_word[1:].isupper():
                    CAMELCASE = 1

                    

    # Find features in the current block of citations, where a block is
    # something like "(Foo 2000; Bar 2011)". Counts the number of citations in
    # various contexts
    all_citations_in_sentence = re.findall("(\(.*?\))", sent)
    for citation_in_sentence in all_citations_in_sentence:
        if citing_string in citation_in_sentence:
            citations = re.findall("([0-9][0-9][0-9][0-9])", citation_in_sentence)
            #print 'Saw %d citations in %s' % (len(citations), citation_in_sentence)
            NUM_CITATIONS_IN_SAME_CITATION = len(citations)

    num_citations_in_sent = len(re.findall("([0-9][0-9][0-9][0-9])", sent))
    #print 'Saw %d citations in sentence: %s' % (num_citations_in_sent, sent)

    # Determine the current subsection and see how many citations are in it
    citance_section_index = sent_index_to_section_id[idx]
    subsec_start_index = citance_section_index
    subsec_end_index = citance_section_index
    for i in range(subsec_start_index, 0, -1):
        if sent_index_to_section_id[i] == citance_section_index:
            subsec_start_index = i
        else:
            break
    for i in range(subsec_end_index, len(sent_index_to_section_id)):
        if sent_index_to_section_id[i] == citance_section_index:
            subsec_end_index = i
        else:
            break
    num_citations_in_subsec = \
        len(re.findall("([0-9][0-9][0-9][0-9])", \
                           ' '.join(text_sents[subsec_start_index:subsec_end_index+1])))


    # Look at the +/-4 tokens right before the citation to see if we find any
    # formulaic patterns
    if len(preceding_text) > 2:
        processed_prec = process_phrase(preceding_text)
    else:
        processed_prec = []
    if len(anteceding_text) > 2:
        processed_ante = process_phrase(anteceding_text)
    else:
        processed_ante = []

    processed_prec = processed_prec[-4:]
    processed_ante = processed_ante[0:4]
    
    prec_formulaic = {}
    prec_action = {}
    if len(processed_prec) > 0:
        prec_formulaic = get_formulaic_features(processed_prec, len(processed_prec), prefix='Preceding:')
        prec_action = get_action_features(processed_prec, len(processed_prec), 'Preceding:')
        pass

    ante_formulaic = {}    
    ante_action = {}    
    if len(processed_ante) > 0:
        ante_formulaic = get_formulaic_features(processed_ante, -1, prefix='Following:')
        ante_action = get_action_features(processed_ante, -1, 'Following:')

                   
    # debug_s = 'We discuss the point of this paper'
        
    # print '\n\n~~~~~~~~~~~\nDEBUG'
    # debug_ps = process_sent(debug_s)
    # formulaic_features = get_formulaic_features(debug_ps) #, debug_s)
    # action_features = get_action_features(debug_ps, '') #, debug_s)
    # print formulaic_features
    # print action_features
    # print 'DEBUG\n\n~~~~~~~~~~~'

    # if debug_label is not None and debug_label.startswith('Prior'):
    #    print 'PRIOR: ' + text_sents[idx]


    # Search for compare and contrast stuff
    DISTANCE_TO_CONTRAST = 100
    DISTANCE_TO_COMPARE = 100
    if True:
        for fw_idx in range(idx+1, min(idx+5, len(text_sents))):
            if fw_idx in parsed_text_sents:
                tmpsent = parsed_text_sents[fw_idx]
            else:
                tmpsent = process_sent(text_sents[fw_idx])
                parsed_text_sents[fw_idx] = tmpsent
            if contains_contrast(tmpsent) and DISTANCE_TO_CONTRAST == 100:
                DISTANCE_TO_CONTRAST = fw_idx - idx
            if contains_compare(tmpsent) and DISTANCE_TO_COMPARE == 100:
                DISTANCE_TO_COMPARE = fw_idx - idx
            # Stop early if both found
            if DISTANCE_TO_COMPARE < 100 and DISTANCE_TO_COMPARE < 100:
                break



    # Parse the current sentence and the citation
    

    #processed_sent = process_sent(sent.replace(citingString, "CITATION"))
    replaced_cite_sent = preceding_text + ' ' + CITATION + ' ' + anteceding_text
    #print 'CONVERTING:\n\t%s\n\t%s\n' % (sent, replaced_cite_sent)
    processed_sent = process_sent(replaced_cite_sent)

    #processed_cite = process_cite(citing_string)
    
    # Figure out where the citation occurs in the parsing
    cite_index = -1
    relative_clause_cite_position = citePositionInSentence
    sent_length = len(processed_sent)
    for ti, token in enumerate(processed_sent):
        if token['word'].endswith('CITATION'):
            cite_index = ti
            break
    if cite_index >= 0:
        if 'segment_span' not in processed_sent[cite_index]:            
            print sent
            print '{"foo": ' + json.dumps(processed_sent) + "}"
            print cite_index
        else:
            span = processed_sent[cite_index]['segment_span']
            clause = processed_sent[span[0]:span[1]]
            #print 'Found %s at %d,\n\tusing "%s"\n\tinstead of "%s"' \
            #    % (citing_string, cite_index, to_str(clause), \
            #           to_str(processed_sent))
            processed_sent = clause
            relative_clause_cite_position = (cite_index - span[0]) / float(span[1] - span[0])
    else:
        #print "couldn't find '%s' in '%s'" \
        #    % (to_str(processed_cite), to_str(processed_sent))
        pass
    clause_length = len(processed_sent)        


    # TODO: find the location of the citation in the processed sentence and then
    # use only the segment of the sentence containing it!
    


    formulaic_features = get_formulaic_features(processed_sent, cite_index)
    agent_features = get_agent_features(processed_sent, cite_index)
    action_features = get_action_features(processed_sent, cite_index, '')
    concept_features = get_concept_features(processed_sent, cite_index, '')


    # Use the 2 sentences before as pre-citance context
    cur_section = sent_index_to_section_id[idx]
    preceding_sents = []
    for prev in range(idx-1, idx-3, -1):
        if prev < 0 or cur_section != sent_index_to_section_id[prev]:
            break
        if prev in parsed_text_sents:
            prec_sent = parsed_text_sents[prev]
        else:
            prec_sent = process_sent(text_sents[prev])            
            parsed_text_sents[prev] = prec_sent
        preceding_sents.append(prec_sent)

    # Use the 4 sentences before as pre-citance context
    following_sents = []
    for foll in range(idx+1, idx+5):
        if foll >= len(text_sents) \
                or cur_section != sent_index_to_section_id[foll]:
            break
        if foll in parsed_text_sents:
            foll_sent = parsed_text_sents[foll]
        else:
            foll_sent = process_sent(text_sents[foll])            
            parsed_text_sents[foll] = foll_sent
        following_sents.append(foll_sent)

    custom_pattern_features = get_custom_pattern_features(\
        processed_sent, preceding_sents, following_sents, cite_index)


    connector_features = get_connector_words(idx, text_sents)

    # Figure out where this citation occurs in the section and subsetion,
    # relatively
    cur_section_id = sent_index_to_section_id[idx] # counts subsections
    cur_section_label = section_id_to_label[cur_section_id] 

    prev_sent_in_subsec = idx
    for prev_sent_in_subsec in range(idx, 0, -1):
        if cur_section_id != sent_index_to_section_id[prev_sent_in_subsec]:
            prev_sent_in_subsec -= 1
            break
    next_sent_in_subsec = idx
    for next_sent_in_subsec in range(idx, len(sent_index_to_section_id)):
        if cur_section_id != sent_index_to_section_id[next_sent_in_subsec]:
            next_sent_in_subsec -= 1
            break
    prev_sent_in_sec = idx
    for prev_sent_in_sec in range(idx, 0, -1):
        if cur_section_id != section_id_to_label[sent_index_to_section_id[prev_sent_in_subsec]]:
            prev_sent_in_sec -= 1
            break
    next_sent_in_sec = idx
    for next_sent_in_sec in range(idx, len(sent_index_to_section_id)):
        if cur_section_label != section_id_to_label[sent_index_to_section_id[next_sent_in_subsec]]:
            next_sent_in_sec -= 1
            break

    relative_pos_in_section = 0
    denom = next_sent_in_sec - prev_sent_in_sec
    if denom > 0:
        relative_pos_in_section = (idx - prev_sent_in_sec) / float(denom)
    relative_pos_in_subsection = 0
    denom = next_sent_in_subsec - prev_sent_in_subsec
    if denom > 0:
        relative_pos_in_subsection = (idx - prev_sent_in_subsec) / float(denom)


    ret_val = OrderedDict()
    ret_val.update(get_section_title_feature( \
            section_id_to_label[sent_index_to_section_id[idx]]))
    ret_val['Section_Num'] = sent_index_to_section_id[idx]
    ret_val['sentence_length'] = sent_length
    ret_val['relative_pos_in_subsection'] = relative_pos_in_subsection
    ret_val['relative_pos_in_section'] = relative_pos_in_section
    ret_val['clause_length'] = sent_length
    ret_val['IS_USED_AS_TEXT'] = IS_USED_AS_TEXT
    ret_val['is_cite_in_parens'] = IS_CITE_IN_PARENS
    ret_val['DOES_SENTENCE_START_WITH_CITATION'] = DOES_SENTENCE_START_WITH_CITATION
    ret_val['NUM_CITATIONS_IN_CITATION_SENTENCE'] = num_citations_in_sent
    ret_val['NUM_CITATIONS_IN_SUBSECTION'] = num_citations_in_subsec
    # ret_val['NUM_CITATIONS_IN_CONTEXT'] = NUM_CITATIONS_IN_CONTEXT
    ret_val['NUM_CITATIONS_IN_SAME_CITATION'] = NUM_CITATIONS_IN_SAME_CITATION


    #ret_val['NUM_NUMBERS_IN_CITATION_SENTENCE'] = NUM_NUMBERS_IN_CITATION_SENTENCE
    #ret_val['NUM_NUMBERS_IN_CONTEXT'] = NUM_NUMBERS_IN_CONTEXT
    #ret_val['NUM_NON_LTW_PRE'] = NUM_NON_LTW_PRE
    #ret_val['NUM_NON_LTW_POST'] = NUM_NON_LTW_POST

    ret_val['CitePositionInSentence'] = citePositionInSentence
    ret_val['relative_clause_cite_position'] = relative_clause_cite_position        
    ret_val['RelativeSentPosition'] = SENTENCE_INDEX / float(len(text_sents))

    # ret_val['DISTANCE_TO_PREVIOUS_NON_CITATION_SENTENCE'] = DISTANCE_TO_PREVIOUS_NON_CITATION_SENTENCE
    ret_val['ALL_CAPS'] = ALL_CAPS
    ret_val['IS_CAPITALIZED'] = IS_CAPITALIZED
    ret_val['CAMELCASE'] = CAMELCASE
    ret_val['IS_EXAMPLE'] = IS_EXAMPLE
    ret_val['IS_CONJUNCTION'] = IS_CONJUNCTION
    ret_val['DISTANCE_TO_CONTRAST'] = DISTANCE_TO_CONTRAST
    ret_val['DISTANCE_TO_COMPARE'] = DISTANCE_TO_COMPARE
    
    if is_self_cite:
        ret_val['IS_SELF_CITE'] = 1

    ret_val.update(formulaic_features)
    ret_val.update(agent_features)
    ret_val.update(action_features)
    ret_val.update(connector_features)
    ret_val.update(custom_pattern_features)                        

    ret_val.update(prec_formulaic)
    ret_val.update(ante_formulaic)
    ret_val.update(prec_action)
    ret_val.update(ante_action)

    if False and debug_label is not None:
        print '%s, %s:\n\tsentece: %s\n\tclause: %s' % (debug_label, citing_string, sent, to_str(processed_sent))
        for k, v in ret_val.iteritems():
            f = float(v)
            z = int(v)
            s = str(v)
            if s != '0' and  z != 100:
                print '\t%s\t%s' % (k, s)
        if ret_val['DISTANCE_TO_CONTRAST'] < 100:
            print '\tSentence with CONTRAST: ' + text_sents[idx+ret_val['DISTANCE_TO_CONTRAST']]
        if ret_val['DISTANCE_TO_COMPARE'] < 100:
            print '\tSentence with COMPARE: ' + text_sents[idx+ret_val['DISTANCE_TO_COMPARE']]
        print ''

    # print sent
    # print ret_val
    return ret_val

# Given the context around the citation (found by ParsCit), determines where
# exactly this citation occurs
def find_citation(parscit_context, text_sents, entire_text, citing_string):
    
    orig_citing_string = citing_string
    # This lets us be robust to the unfortunate tokenization that CoreNLP does
    # to citation texts :/
    #citing_string = reformat_cite(citing_string)

    sentence_end_offsets = []
    len_sum = 0
    for sent in text_sents:
        # The +1 is for the space during join
        len_sum += len(sent) + 1
        sentence_end_offsets.append(len_sum)
    

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    context_sents = sent_detector.tokenize(parscit_context.decode('utf-8', 'ignore'))

    try:
        pass
        #context_sents = []
        #processed_context_sents = json.loads(corenlp.annotate(parscit_context.encode('utf-8'), \
        #    properties={'annotators': 'tokenize,ssplit' }).encode('utf-8'), strict=False)
        #for sent in processed_context_sents['sentences']:
        #    sent_text = to_str(sent['tokens']).encode('utf-8')
        #    context_sents.append(sent_text)

    except ValueError as e:
        # This happens sometimes when Parscit encodes weird codepoints, which
        # break both NLTK and CoreNLP :(
        print repr(e)
        # Indicate no context could be found
        return -1

    

    # cite_offset = entire_text.find(parscit_context)
    # print 'cite offset: %d' % (cite_offset)
    best_match = -1
    best_str = -1
    best_offset = -1

    context_len = len(parscit_context)
    cur_find_index = -1;
    pre_context = parscit_context.find(citing_string)
    # This is weird but happens due to mangling
    if pre_context < 0:
        pre_context = len(parscit_context) / 2
    post_context = len(parscit_context) - len(citing_string) - pre_context - 1
    
    while True:
        cur_find_index = entire_text.find(citing_string, cur_find_index+1)
        if cur_find_index < 0:
            break
        matches = 0
        
        s = SequenceMatcher(None, entire_text[cur_find_index-pre_context:cur_find_index+post_context], parscit_context)
        quick_ratio = s.real_quick_ratio()
        if quick_ratio > best_match:
            ratio = s.quick_ratio()
            if best_match < ratio:
                best_match = ratio
                best_offset = cur_find_index-pre_context
    
    #print 'Found the best offset at %d with match ratio %d' % (best_offset, best_match)

    # Once we have a rough match, figure out which sentence this is
    
    sent_index_of_context_start = -1
    for idx, offset in enumerate(sentence_end_offsets):
        if best_offset < offset:
            sent_index_of_context_start = idx
            break
    # print 'Figured out that parcit context starts at sentence %d' % (sent_index_of_context_start)

    # Figure out which sentence in the context has the citation, which should be
    # in the middle-ish
    num_context_sents = len(context_sents) + 2
    middle_index = num_context_sents/2 + sent_index_of_context_start
    citing_sentence_index = -1
    if middle_index >= len(text_sents):
        middle_index = len(text_sents) - 1
    
    #print 'searching for "%s" in indices [%d, %d]' % \
    #    (citing_string, sent_index_of_context_start, \
    #         sent_index_of_context_start + num_context_sents)

    for i in range(0, (num_context_sents / 2) + 1):
        #print 'i: %d, mi: %d, mi - i: %d, len(sents): %d' \
        #    % (i, middle_index, middle_index - 1, len(text_sents))
        # Search from middle out
        if middle_index + i < len(text_sents) and citing_string in text_sents[middle_index + i]:
            citing_sentence_index = middle_index + i
            break
        elif middle_index - i >= 0 and citing_string in text_sents[middle_index - i]:
            citing_sentence_index = middle_index - i
            break

    #citing_sentence = None
    #if citing_sentence_index >= 0:
    #    citing_sentence = text_sents[citing_sentence_index]

    return citing_sentence_index



def get_number_of_non_ltw_chars(text):
    return len(text) - sum([x.isalnum() or x.isspace() for x in text])
    
def get_all_citation_strings(xmldoc):
    all_citations = []

    itemlist = xmldoc.getElementsByTagName('citation')
    for item in itemlist:
        parscit_contexts = item.getElementsByTagName("context")
        for parscit_context in parscit_contexts:
            citeTxt = parscit_context.firstChild.data
            citingString = parscit_context.attributes['citStr'].value
            all_citations.append(citingString)
    return set(all_citations)

def check_example_terms(sent):
    terms = ["example", "such as", "see", "e.g."]
    for term in terms:
        if term in sent:
            return 1
    return 0

def get_number_of_citations_in_text(sent, xmldoc):
    all_citations = get_all_citation_strings(xmldoc)
    count = 0
    for citation in all_citations:
        if citation in sent:
            count += 1
    return count

def get_citation_repetition(sent, citation):
    return sent.count(citation)

def get_number_of_numbers_in_text(sent):
    import re
    return len(re.findall("[\d.]+", sent))

def getPaperText(xmldoc):
    bodytextlist = xmldoc.getElementsByTagName('bodyText')
    ALLTEXT = ''
    for bodytext in bodytextlist:
        TEXT = bodytext.firstChild.data.strip()
        TEXT = " ".join(TEXT.splitlines())
        TEXT = TEXT.replace("- ", "")
        ALLTEXT += TEXT

    return ALLTEXT

def getPaperSentences(xmldoc):
    bodytextlist = xmldoc.getElementsByTagName('bodyText')
    ALLTEXT = ''

    for bodytext in bodytextlist:
        TEXT = bodytext.firstChild.data.strip()
        TEXT = " ".join(TEXT.splitlines())
        TEXT = TEXT.replace("- ", "")
        ALLTEXT += TEXT

    ALLTEXT = ALLTEXT.replace("cf.", "cfXXX")

    # Strip out the manged ACL footers

    # 177 Computational Linguistics Volume 17, Number 2
    ALLTEXT = re.sub(r'[0-9]*\s+Computational\s+Linguistics\s+Volume\s+[0-9]+\s*,\s+Number\s+[0-9]+',
           ' ', ALLTEXT)

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text_sents = sent_detector.tokenize(ALLTEXT)

    to_return_text_sents = []
    for sent in text_sents:
        sent = sent.replace("cfXXX", "cf.")
        to_return_text_sents.append(sent)
    
    return to_return_text_sents

def print_node(root, ALLTEXT):
    if root.childNodes:
        for node in root.childNodes:
           if node.nodeType == node.ELEMENT_NODE:
               if node.tagName in ['bodyText', 'listItem', 'footnote'] or 'Header' in node.tagName:
                   TEXT = node.firstChild.data.strip()
                   TEXT = " ".join(TEXT.splitlines())
                   TEXT = TEXT.replace("- ", "")
                   ALLTEXT += " " + TEXT
               ALLTEXT = print_node(node, ALLTEXT)
    return ALLTEXT

def normalize_string(ALLTEXT):
    ALLTEXT = ALLTEXT.replace("cf.", "cfXXX")
    ALLTEXT = ALLTEXT.replace("??", "XX")
    #ALLTEXT = ALLTEXT.replace("et al.", "et al")
    return ALLTEXT

def lemmatize(text):
#    tokens = word_tokenize(text)
    lemmas = []
    try:
        output = json.loads(corenlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,lemma' }), strict=False)
        if 'sentences' in output:
            for sent in output['sentences']:
                if 'tokens' in sent:
                    for t in sent['tokens']:
                        lemmas.append(t['lemma'])
    # Some kind of JSON Exception
    except ValueError as e:
        pass
    lemmatized = ' '.join(lemmas)
    #print "LEMMAS: %s -> %s" % (text, lemmatized)
    return lemmatized

def parse(text):
#    tokens = word_tokenize(text)
    lemmas = []
    try:
        output = json.loads(corenlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }), strict=False)
        if 'sentences' in output:
            for sent in output['sentences']:
                if 'tokens' in sent:
                    for t in sent['tokens']:
                        lemmas.append(t['lemma'])
    # Some kind of JSON Exception
    except ValueError as e:
        pass
    lemmatized = ' '.join(lemmas)
    #print "LEMMAS: %s -> %s" % (text, lemmatized)
    return lemmatized




def getPaperSentences_2(xmldoc):

    ALLTEXT = print_node(xmldoc.documentElement, "")
    ALLTEXT = normalize_string(ALLTEXT)

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text_sents = sent_detector.tokenize(ALLTEXT)

    '''to_return_text_sents = []
    for sent in text_sents:
        sent = sent.replace("cfXXX", "cf.")
        to_return_text_sents.append(sent)'''

    return text_sents



CONNECTORS = ["therefore", "however", "moreover", "in addition", "furthermore", "so" ,"also", "likewise", "accordingly", "hence", "consequent", "as a result", "therewise", "otherwise", "subsequently", "lastly", "next", "thus", "then", "wherefore", "similarly", "particular", "for instance" ,"special", "for example", "compare", "beside", "conclusion" ,"summar", "result", "analog"]


def get_connector_words(idx, text_sents):
    seen_connectors = {}

    for c in CONNECTORS:
        seen_connectors[c] = 100

    for fw_idx in range(idx, min(idx+8, len(text_sents))):

        sent = text_sents[fw_idx].lower()
        for c in CONNECTORS:
            if sent.startswith(c):
                seen_connectors[c] = (fw_idx - idx) + 1

    return seen_connectors


ALL_ACTION_LEXICONS = {
    "AFFECT": ["afford", "believe", "decide", "feel", "hope", "imagine", "regard", "trust", "think"], 

    "ARGUMENTATION": ["agree", "accept", "advocate", "argue", "claim", "conclude", "comment", "defend", "embrace", "hypothesize", "imply", "insist", "posit", "postulate", "reason", "recommend", "speculate", "stipulate", "suspect"],

    "AWARE": ["be unaware", "be familiar with", "be aware", "be not aware", "know of"],

    "BETTER_SOLUTION": ["boost", "enhance", "defeat", "improve", "go beyond", "perform better", "outperform", "outweigh", "surpass"],
    
    "CHANGE": ["adapt", "adjust", "augment", "combine", "change", "decrease", "elaborate", "expand", "expand on", "extend", "derive", "incorporate", "increase", "manipulate", "modify", "optimize", "optimise", "refine", "render", "replace", "revise", "substitute", "tailor", "upgrade"], 
         
    "COMPARISON": ["compare", "compete", "evaluate", "test"],
         
    "DENOTATION": ["be", "denote", "represent" ],

    "INSPIRATION": ["inspire", "motivate" ],

    "AGREE": ["agree with", "side with" ],

    "CONTINUE": ["adopt", "base", "be base on", 'base on', "derive from", "originate in",  "borrow", "build on", "follow", "following", "originate from", "originate in", 'start from', 'proceed from'],

    "CONTRAST": ["be different from", "be distinct from", "conflict", "contrast", "clash", "differ from", "distinguish", "differentiate", "disagree", "disagreeing", "dissent", "oppose"],

    "FUTURE_INTEREST": [ "be interest in", "plan on", "plan to", "expect to", "intend to", "hope to"],

    "HEDGING_MODALS": ["could", "might", "may", "should" ],

    "FUTURE_MODALS": ["will", "going to" ],

    "SHOULD": ["should" ],

    "INCREASE": ["increase", "grow", "intensify", "build up", "explode" ],

    "INTEREST": ["aim", "ask", "address", "attempt", "be concern", "be interest", "be motivat", "concern", "concern", "concern", "consider", "concentrate on", "explore", "focus", "intend to", "like to", "look at how", "pursue", "seek", "study", "try", "target", "want", "wish", "wonder"],

    "NEED": ["be dependent on", "be reliant on", "depend on", "lack", "need", "necessitate", "require", "rely on"],

    "PRESENTATION": ["describe", "discuss", "give", "introduce", "note", "notice", "point out", "present", "propose", "put forward", "recapitulate", "remark", "report", "say", "show", "sketch", "state", "suggest", "talk about"], 

    "PROBLEM": ["abound", "aggravate", "arise", "be cursed", "be incapable of", "be force to", "be limite to", "be problematic", "be restrict to", "be trouble", "be unable to", "contradict", "damage", "degrade", "degenerate", "fail", "fall prey", "fall short", "force", "force", "hinder", "impair", "impede", "inhibit", "misclassify", "misjudge", "mistake", "misuse", "neglect", "obscure", "overestimate", "over-estimate", "overfit", "over-fit", "overgeneralize", "over-generalize", "overgeneralise", "over-generalise", "overgenerate", "over-generate", "overlook", "pose", "plague", "preclude", "prevent", "remain", "resort to", "restrain", "run into", "settle for", "spoil", "suffer from", "threaten", "thwart", "underestimate", "under-estimate", "undergenerate", "under-generate", "violate", "waste", "worsen"], 
        
    "RESEARCH": ["apply", "analyze", "analyse", "build", "calculate", "categorize", "categorise", "characterize", "characterise", "choose", "check", "classify", "collect", "compose", "compute", "conduct", "confirm", "construct", "count", "define", "delineate", "design", "detect", "determine", "equate", "estimate", "examine", "expect", "formalize", "formalise", "formulate", "gather", "identify", "implement", "indicate", "inspect", "integrate", "interpret", "investigate", "isolate", "maximize", "maximise", "measure", "minimize", "minimise", "observe", "predict", "realize", "realise", "reconfirm", "revalidate", "simulate", "select", "specify", "test", "verify", "work on"], 

    "SEE": [ "see", "view", "treat", "consider" ],

    "SIMILAR": ["bear comparison", "be analogous to", "be alike", "be related to", "be closely relate to", "be reminiscent of", "be the same as", "be similar to", "be in a similar vein to", "have much in common with", "have a lot in common with", "pattern with", "resemble"],

    "SOLUTION": ["accomplish", "account for", "achieve", "apply to", "answer", "alleviate", "allow for", "allow", "allow", "avoid", "benefit", "capture", "clarify", "circumvent", "contribute", "cope with", "cover", "cure", "deal with", "demonstrate", "develop", "devise", "discover", "elucidate", "escape", "explain", "fix", "gain", "go a long way", "guarantee", "handle", "help", "implement", "justify", "lend itself", "make progress", "manage", "mend", "mitigate", "model", "obtain", "offer", "overcome", "perform", "preserve", "prove", "provide", "realize", "realise", "rectify", "refrain from", "remedy", "resolve", "reveal", "scale up", "sidestep", "solve", "succeed", "tackle", "take care of", "take into account", "treat", "warrant", "work well", "yield"],

    "TEXTSTRUCTURE": ["begin by", "illustrate", "conclude by", "organize", "organise", "outline", "return to", "review", "start by", "structure", "summarize", "summarise", "turn to"], 
         
    "USE": ["apply", "employ", "use", "make use", "utilize", "implement", 'resort to']
    }
# Split the patterns into lists of things to match
for feature_name in list(ALL_ACTION_LEXICONS.keys()):
    tokenized = []
    patterns = ALL_ACTION_LEXICONS[feature_name]
    for pattern in patterns:
        tokenized.append(pattern.split())
    ALL_ACTION_LEXICONS[feature_name] = tokenized


ALL_CONCEPT_LEXICONS = {
    "NEGATION": ["no", "not", "nor", "non", "neither", "none", "never", "aren't", "can't", "cannot", "hadn't", "hasn't", "haven't", "isn't", "didn't", "don't", "doesn't", "n't", "wasn't", "weren't", "nothing", "nobody", "less", "least", "little", "scant", "scarcely", "rarely", "hardly", "few", "rare", "unlikely"],
    "3RD_PERSON_PRONOUN_(NOM)": ["they", "he", "she", "theirs", "hers", "his"],
    "OTHERS_NOM": ["they", "he", "she", "theirs", "hers", "his"],
    "3RD_PERSON)PRONOUN_(ACC)": ["her", "him", "them"], 
    "OTHERS_ACC": ["her", "him", "them"], 
    "3RD_POSS_PRONOUN": ["their", "his", "her"],
    #"OTHERS_POSS": ["their", "his", "her"],
    "OTHERS_POSS": ["their", "his", "her", "they"],
    "3RD_PERSON_REFLEXIVE": ["themselves", "himself", "herself"],
    "1ST_PERSON_PRONOUN_(NOM)": ["we", "i", "ours", "mine"],
    "SELF_NOM": ["we", "i", "ours", "mine"],
    "1ST_PERSON_PRONOUN_(ACC)": ["us", "me"],
    "SELF_ACC": ["us", "me"],
    "1ST_POSS_PRONOUN": ["my", "our"],
    "SELF_POSS": ["my", "our"],
    "1ST_PERSON_REFLEXIVE ": ["ourselves", "myself"],
    "REFERENTIAL": ["this", "that", "those", "these"],
    "REFLEXIVE": ["itself ourselves", "myself", "themselves", "himself", "herself"],
    "QUESTION": ["?", "how", "why", "whether", "wonder"],
    "GIVEN": ["noted", "mentioned", "addressed", "illustrated", "described", "discussed", "given", "outlined", "presented", "proposed", "reported", "shown", "taken"],
    
    "PROFESSIONALS": ["collegues", "community", "computer scientists", "computational linguists", "discourse analysts", "expert", "investigators", "linguists", "logicians", "philosophers", "psycholinguists", "psychologists", "researchers", "scholars", "semanticists", "scientists"],

    "DISCIPLINE": ["computerscience", "computer linguistics", "computational linguistics", "discourse analysis", "logics", "linguistics", "psychology", "psycholinguistics", "philosophy", "semantics", "lexical semantics", "several disciplines", "various disciplines"],
    
    "TEXT_NOUN": ["paragraph", "section", "subsection", "chapter"],
    
    "SIMILAR_NOUN": ["analogy", "similarity"],

    "SIMILAR_ADJ": ["similar", "comparable", "analogous", "kindred"],

    "COMPARISON_NOUN": ["accuracy", "baseline", "comparison", "competition", "evaluation", "inferiority", "measure", "measurement", "performance", "precision", "optimum", "recall", "superiority"],
    
    "CONTRAST_NOUN": ["contrast", "conflict", "clash", "clashes", "difference", "point of departure"],

    "AIM_NOUN": ["aim", "direction", "goal", "intention", "objective", "purpose", "task", "theme", "topic"],

    "ARGUMENTATION_NOUN": ["assumption", "belief", "hypothesis", "hypotheses", "claim", "conclusion", "confirmation", "opinion", "recommendation", "stipulation", "view"],

    "PROBLEM_NOUN": ["Achilles heel", "caveat", "challenge", "complication", "contradiction", "damage", "danger", "deadlock", "defect", "detriment", "difficulty", "dilemma", "disadvantage", "disregard", "doubt", "downside", "drawback", "error", "failure", "fault", "foil", "flaw", "handicap", "hindrance", "hurdle", "ill", "inflexibility", "impediment", "imperfection", "intractability", "inefficiency", "inadequacy", "inability", "lapse", "limitation", "malheur", "mishap", "mischance", "mistake", "obstacle", "oversight", "pitfall", "problem", "shortcoming", "threat", "trouble", "vulnerability", "absence", "dearth", "deprivation", "lack", "loss", "fraught", "proliferation", "spate"],

    "QUESTION_NOUN": ["question", "conundrum", "enigma", "paradox", "phenomena", "phenomenon", "puzzle", "riddle"],

    "SOLUTION_NOUN": ["answer", "accomplishment", "achievement", "advantage", "benefit", "breakthrough", "contribution", "explanation", "idea", "improvement", "innovation", "insight", "justification", "proposal", "proof", "remedy", "solution", "success", "triumph", "verification", "victory"],

    "INTEREST_NOUN": ["attention", "quest"],

    # Not sure if this one is used
    "RESEARCH_NOUN": ["evidence", "experiment", "finding", "progress", "observation", "outcome", "result"],

    "RESULT_NOUN": ["evidence", "experiment", "finding", "progress", "observation", "outcome", "result"],

    "METRIC_NOUN": ["bleu", "F-score", "F1-score", "F score", "F1 score", "precision", "recall", "accuracy", "correlation"],

    "CHANGE_NOUN": [ "adaptation", "enhancement", "extension", "generalization", "development", "modification", "refinement", "version", "variant", "variation"],

    "PRESENTATION_NOUN": ["article", "draft", "manuscript", "paper", "project", "report", "study"],
    
    "NEED_NOUN": ["necessity", "motivation"],

    "WORK_NOUN": ["account", "algorithm", "analysis", "analyses", "approach", "approaches", "application", "architecture", "characterization", "characterisation", "component", "design", "extension", "formalism", "formalization", "formalisation", "framework", "implementation", "investigation", "machinery", "method", "methodology", "model", "module", "moduls", "process", "procedure", "program", "prototype", "research", "researches", "strategy", "system", "technique", "theory", "tool", "treatment", "work"],

    "TRADITION_NOUN": ["acceptance", "community", "convention", "disciples", "disciplines", "folklore", "literature", "mainstream", "school", "tradition", "textbook"],

    "CHANGE_ADJ": ["alternate", "alternative"],
    
    "GOOD_ADJ": ["adequate", "advantageous", "appealing", "appropriate", "attractive", "automatic", "beneficial", "capable", "cheerful", "clean", "clear", "compact", "compelling", "competitive", "comprehensive", "consistent", "convenient", "convincing", "constructive", "correct", "desirable", "distinctive", "efficient", "effective", "elegant", "encouraging", "exact", "faultless", "favourable", "feasible", "flawless", "good", "helpful", "impeccable", "innovative", "insightful", "intensive", "meaningful", "neat", "perfect", "plausible", "positive", "polynomial", "powerful", "practical", "preferable", "precise", "principled", "promising", "pure", "realistic", "reasonable", "reliable", "right", "robust", "satisfactory", "simple", "sound", "successful", "sufficient", "systematic", "tractable", "usable", "useful", "valid", "unlimited", "well worked out", "well", "enough", "well-motivated"],
    
    "BAD_ADJ": ["absent", "ad-hoc", "adhoc", "ad hoc", "annoying", "ambiguous", "arbitrary", "awkward", "bad", "brittle", "brute-force", "brute force", "careless", "confounding", "contradictory", "defect", "defunct", "disturbing", "elusive", "erraneous", "expensive", "exponential", "false", "fallacious", "frustrating", "haphazard", "ill-defined", "imperfect", "impossible", "impractical", "imprecise", "inaccurate", "inadequate", "inappropriate", "incomplete", "incomprehensible", "inconclusive", "incorrect", "inelegant", "inefficient", "inexact", "infeasible", "infelicitous", "inflexible", "implausible", "inpracticable", "improper", "insufficient", "intractable", "invalid", "irrelevant", "labour-intensive", "laborintensive", "labour intensive", "labor intensive", "laborious", "limited-coverage", "limited coverage", "limited", "limiting", "meaningless", "modest", "misguided", "misleading", "nonexistent", "NP-hard", "NP-complete", "NP hard", "NP complete", "questionable", "pathological", "poor", "prone", "protracted", "restricted", "scarce", "simplistic", "suspect", "time-consuming", "time consuming", "toy", "unacceptable", "unaccounted for", "unaccounted-for", "unaccounted", "unattractive", "unavailable", "unavoidable", "unclear", "uncomfortable", "unexplained", "undecidable", "undesirable", "unfortunate", "uninnovative", "uninterpretable", "unjustified", "unmotivated", "unnatural", "unnecessary", "unorthodox", "unpleasant", "unpractical", "unprincipled", "unreliable", "unsatisfactory", "unsound", "unsuccessful", "unsuited", "unsystematic", "untractable", "unwanted", "unwelcome", "useless", "vulnerable", "weak", "wrong", "too", "overly", "only"],

    "BEFORE_ADJ": ["earlier", "initial", "past", "previous", "prior"],
    
    "CONTRAST_ADJ": ["different", "distinguishing", "contrary", "competing", "rival"],

    "CONTRAST_ADV": ["differently", "distinguishingly", "contrarily", "otherwise", "other than", "contrastingly", "imcompatibly", "on the other hand", ],

    "TRADITION_ADJ": ["better known", "better-known", "cited", "classic", "common", "conventional", "current", "customary", "established", "existing", "extant", "available", "favourite", "fashionable", "general", "obvious", "long-standing", "mainstream", "modern", "naive", "orthodox", "popular", "prevailing", "prevalent", "published", "quoted", "seminal", "standard", "textbook", "traditional", "trivial", "typical", "well-established", "well-known", "widelyassumed", "unanimous", "usual"],

    "MANY": ["a number of", "a body of", "a substantial number of", "a substantial body of", "most", "many", "several", "various"],

    "HELP_NOUN": ['help', 'aid', 'assistance', 'support' ],

    "SENSE_NOUN": ['sense', 'spirit', ],

    "GRAPHIC_NOUN": ['table', 'tab', 'figure', 'fig', 'example' ],
    
    "COMPARISON_ADJ": ["evaluative", "superior", "inferior", "optimal", "better", "best", "worse", "worst", "greater", "larger", "faster", "weaker", "stronger"],

    "PROBLEM_ADJ": ["demanding", "difficult", "hard", "non-trivial", "nontrivial"],
    
    "RESEARCH_ADJ": ["empirical", "experimental", "exploratory", "ongoing", "quantitative", "qualitative", "preliminary", "statistical", "underway"],

    "AWARE_ADJ": ["unnoticed", "understood", "unexplored"],

    "NEED_ADJ": ["necessary", "indispensable", "requisite"],

    "NEW_ADJ": ["new", "novel", "state-of-the-art", "state of the art", "leading-edge", "leading edge", "enhanced"],

    "FUTURE_ADJ": ["further", "future"],

    "HEDGE_ADJ": [ "possible", "potential", "conceivable", "viable"],
    
    "MAIN_ADJ": ["main", "key", "basic", "central", "crucial", "critical", "essential", "eventual", "fundamental", "great", "important", "key", "largest", "main", "major", "overall", "primary", "principle", "serious", "substantial", "ultimate"],

    "CURRENT_ADV": ["currently", "presently", "at present"],

    "TEMPORAL_ADV": ["finally", "briefly", "next"],

    "SPECULATION": [],

    "CONTRARY": [],

    "SUBJECTIVITY": [],

    "STARSEM_NEGATION": [  "contrary", "without", "n't", "none", "nor", "nothing", "nowhere", "refused", "nobody", "means", "never", "neither", "absence", "except", "rather", "no", "for", "fail", "not", "neglected", "less", "prevent", 
 ],

    'DOWNTONERS': [ 'almost', 'barely', 'hardly', 'merely', 'mildly', 'nearly', 'only', 'partially', 'partly', 'practically', 'scarcely', 'slightly', 'somewhat', ],

    'AMPLIFIERS': [ 'absolutely', 'altogether', 'completely', 'enormously', 'entirely', 'extremely', 'fully', 'greatly', 'highly', 'intensely', 'strongly', 'thoroughly', 'totally', 'utterly', 'very', ],

    
    'PUBLIC_VERBS': ['acknowledge', 'admit', 'agree', 'assert', 'claim', 'complain', 'declare', 'deny', 'explain', 'hint', 'insist', 'mention', 'proclaim', 'promise', 'protest', 'remark', 'reply', 'report', 'say', 'suggest', 'swear', 'write', ],
    
    'PRIVATE_VERBS': [ 'anticipate', 'assume', 'believe', 'conclude', 'decide', 'demonstrate', 'determine', 'discover', 'doubt', 'estimate', 'fear', 'feel', 'find', 'forget', 'guess', 'hear', 'hope', 'imagine', 'imply', 'indicate', 'infer', 'know', 'learn', 'mean', 'notice', 'prove', 'realize', 'recognize', 'remember', 'reveal', 'see', 'show', 'suppose', 'think', 'understand', ],
    
    'SUASIVE_VERBS': [ 'agree', 'arrange', 'ask', 'beg', 'command', 'decide', 'demand', 'grant', 'insist', 'instruct', 'ordain', 'pledge', 'pronounce', 'propose', 'recommend', 'request', 'stipulate', 'suggest', 'urge', ]


    }
# Split the patterns into lists of things to match
for feature_name in list(ALL_CONCEPT_LEXICONS.keys()):
    tokenized = []
    patterns = ALL_CONCEPT_LEXICONS[feature_name]
    for pattern in patterns:
        tokenized.append(pattern.split())
    ALL_CONCEPT_LEXICONS[feature_name] = tokenized


def anywhere_in_lexicon(lexicon, sentence):
    for i in range(0, len(sentence)):
        (is_match, x) = is_in_lexicon(lexicon, sentence, i)
        if is_match:
            return (is_match, x)
    return (False, 0)

def is_in_lexicon(lexicon, sentence, si, ArgType=None, required_pos=None):
    for phrase in lexicon:

        # Can't match phrases that would extend beyond this sentence
        if len(phrase) + si > len(sentence):
            continue

        found = True
        found_arg = False

        for i, lemma in enumerate(phrase):
            # Check the word form too, just to prevent weird lemmatization
            # issues (usually for adjectives)
            if not (sentence[si+i]['lemma'] == lemma or sentence[si+i]['word'] == lemma) \
                    or not (required_pos is None or sentence[si+i]['pos'][0] == required_pos):
                found = False
                break
            if ArgType is not None and sentence[si+i]['ArgType'] == ArgType:
                found_arg = True
        if found and (ArgType is None or found_arg):
            #if len(phrase) > 1:
            #    print '~~~~~~Matched %s' % (' '.join(phrase))
            return (True, len(phrase))
    return (False, 0)
                

def contains_contrast(sentence):
    if len(sentence) < 2:
        return False
    
    if sentence[0]['lemma'] == 'in' and sentence[1]['lemma'] == 'contrast':
        return True
    
    contains_contrast = False

    for pattern in FORMULAIC_PATTERNS['CONTRAST_FORMULAIC']:
        if find(pattern, sentence, None) >= 0:
            contains_contrast = True
            break

    if not contains_contrast:
        for pattern in FORMULAIC_PATTERNS['CONTRAST2_FORMULAIC']:
            if find(pattern, sentence, None) >= 0:
                contains_contrast = True
                break

    if not contains_contrast:
        return False

    # Must contain mention of this work as well
    for pattern in AGENT_PATTERNS['US_AGENT']:
        if find(pattern, sentence, None) >= 0:
            return True

    return False

def contains_compare(sentence):
    if len(sentence) == 0:
        return False
    
    for pattern in FORMULAIC_PATTERNS['COMPARISON_FORMULAIC']:
        #if find(pattern, sentence, None, debug=to_str(sentence), feature='contrast-check'):
        if find(pattern, sentence, None) >= 0:
            return True
    return False
    
def to_str(sentence):
    tokens = []
    for i in range(0, len(sentence)):
        token = sentence[i]['word']
        if token == '-LRB-':
            token = '('
        elif token == '-RRB-':
            token = ')'
        elif token == '-LSB-':
            token = '['
        elif token == '-RSB-':
            token = ']'

        tokens.append(token)
    return ' '.join(tokens)

def get_concept_features(sentence, cite_index, prefix):
    
    features = {} 

    for feature_name, lexicon in ALL_CONCEPT_LEXICONS.iteritems():
        for i in range(0, len(sentence)):            
            (is_match, matched_phrased_length) = is_in_lexicon(lexicon, sentence, i)

            #print 'found %s (%d) in %s? %s (%d)' % (sentence[i]['lemma'], i, feature_name, is_match, matched_phrased_length)
            if is_match:
                features[prefix + feature_name] = i - cite_index
                break

    #print 'Saw action features %s in %s' % (str(features), to_str(sentence))
    return features


def get_action_features(sentence, cite_index, prefix):
    
    features = {} 

    for feature_name, lexicon in ALL_ACTION_LEXICONS.iteritems():
        for i in range(0, len(sentence)):            
            (is_match, matched_phrased_length) = is_in_lexicon(lexicon, sentence, i, required_pos='V')

            #print 'found %s (%d) in %s? %s (%d)' % (sentence[i]['lemma'], i, feature_name, is_match, matched_phrased_length)
            if is_match:
                features[prefix + feature_name] = i - cite_index
                break
    # print 'Saw action features %s in %s' % (str(features), to_str(sentence))
    return features




def find_cite(parsed_sent, parsed_cite):
    for i in range(0, (len(parsed_sent) - len(parsed_cite)) + 1):
        match = True
        for j in range(0, len(parsed_cite)):
            if parsed_sent[i+j]['word'] != parsed_cite[j]['word']:
                match = False
                break
        if match:
            return i
    return -1

def parse_all(sentences):
    parsed = []
    for sent in sentences:
        parsed.append(process_sent(sent))
    return parsed

def process_cite(cite_string):
    tokenized_cite = json.loads(corenlp.annotate(cite_string, properties={'annotators': 'tokenize,ssplit,pos' }), strict=False)
    tokens = tokenized_cite['sentences'][0]['tokens']
    for token in tokens:
        if token['word'] == '-LRB-':
            token['word'] = '('
        elif token['word'] == '-RRB-':
            token['word'] = ')'
        elif token['word'] == '-LSB-':
            token['word'] = '['
        elif token['word'] == '-RSB-':
            token['word'] = ']'

    return tokens


def process_phrase(text):
    output = json.loads(corenlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }), strict=False)
    sentences = output['sentences']
    if len(sentences) > 1:
        sentences = sentences[:1]
        #raise BaseException('ughh too many sentences???')
    sentence = sentences[0]
    # this is what we really want
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

    return tokens

def process_sent(text, get_deps=False):
    output = None
    for i in range(1, 5):
        try:
            output = json.loads(corenlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }), strict=False)
            break
        except ValueError:
            pass

    if output is None:
        if get_deps:
            return [], {}
        else:
            return []
    
    sentences = output['sentences']

    # In the event we see multiple sentences
    if len(sentences) > 1:
        pass
        #print 'SAW MULTI-SENTENCE: ', text
        #for sent in sentences:
            #print '\t', to_str(sent['tokens'])
        #sentences = sentences[:1]
        #raise BaseException('ughh too many sentences???')
    sentence = sentences[0]
    # this is what we really want
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

    #print json.dumps(tokens)

    
    # Break the sentence into clausal regions (dependencies clauses, etc)
    #print '\n', text
    segment(tokens, deps)

    if get_deps:
        return tokens, deps
    else:
        return tokens


def segment(tokens, dep_tree):
    tree = {}

    # Build the tree
    for dep in dep_tree:
        if dep['dep'] == 'ROOT':
            tree['ROOT'] = []
            tree['ROOT'].append(dep['dependent'])
            if dep['dependent'] not in tree:
                tree[dep['dependent']] = []
        else:
            gov_id = dep['governor']
            dep_id = dep['dependent']
            if dep['dep'] == 'punct':
                if dep_id == len(tokens):
                    continue

            if gov_id not in tree:
                tree[gov_id] = []
            if dep_id not in tree:
                tree[dep_id] = []

            tree[gov_id].append(dep_id)
        

    # Find the clausal roots of the tree
    roots = []        
    for dep in dep_tree:
        dep_type = dep['dep']
        if dep_type == 'ROOT' or dep_type == 'advcl' \
                or dep_type == 'ccomp' or dep_type.startswith('acl:') \
                or dep_type == 'list' or dep_type == 'parataxis':
            # Check for the length in case this has no dependent ?!?!
            dep_id = dep['dependent']
            #print '%s -> %s' % (dep_type, str(dep_id))
            if isinstance(dep_id, int):
                roots.append(dep_id)
        # Look for coordinating phrases linked by a verb
        elif dep_type == 'conj' or dep_type == 'cc':
            gov_id = dep['governor']
            dep_id = dep['dependent']
            
            if (tokens[dep_id-1]['pos'][0] == 'V' or tokens[dep_id-1]['pos'] == 'IN') \
                    and (tokens[gov_id-1]['pos'][0] == 'V' \
                             or tokens[gov_id-1]['pos'] == 'IN'):
                #print 'case2: ', dep['dependent']
                #print '%s -> %s' % (dep_type, str(dep['dependent']))
                roots.append(dep['dependent'])
    
    # Walk the tree from the roots, following each link
    #print 'roots: ', roots
    for root in roots:

        #print 'walking %s' % (str(root))
        children = []
        # Sometimes ARC papers have horrifically long sentences, which are not
        # actually from research papers (e.g., the proceedings introduction).
        # In this case, we just skip them and say the span is the entire
        # sentence
        if len(tokens) >= 500:

            span = (0, len(tokens))
            for t in tokens:
                t['segment_span'] = span

            continue

        walk(tree, root, children, roots)
        children.append(root)
        children.sort()          

        if len(children) == 0:
            continue
        
        #print 'Root %d had children: %s' % (root, str(children))

        span = (children[0]-1, children[-1])
        
        for i in range(children[0]-1, children[-1]):
            tokens[i]['segment_span'] = span

        # Find the root and see if it has a Modal
        root_id = root

        # Mark the verb tense
        if tokens[root_id-1]['pos'][0] == 'V':
            tokens[root_id-1]['is_root'] = True

        if tokens[root_id-1]['pos'][0] == 'V':
            # Use the POS tag tenses
            tokens[root_id-1]['tense'] = tokens[root_id-1]['pos'][-1]

        for dep in dep_tree:
            if dep['dep'] == 'aux' and dep['governor'] == root_id:
                tokens[root_id-1]['has_aux'] = True
            if dep['dep'] == 'auxpass' and dep['governor'] == root_id:
                tokens[root_id-1]['is_pass'] = True

        


def walk(tree, cur, seen, exclude):
    for child in tree[cur]:
        if child in exclude:
            continue
        seen.append(child)
        walk(tree, child, seen, exclude)




FORMULAIC_PATTERNS = {
    'GENERAL_FORMULAIC': [ 'in @TRADITION_ADJ #JJ @WORK_NOUN',
                           'in @TRADITION_ADJ used @WORK_NOUN',
                           'in @TRADITION_ADJ @WORK_NOUN',
                           'in @MANY #JJ @WORK_NOUN',
                           'in @MANY @WORK_NOUN',
                           'in @BEFORE_ADJ #JJ @WORK_NOUN',
                           'in @BEFORE_ADJ @WORK_NOUN',
                           'in other #JJ @WORK_NOUN',
                           #'in other @WORK_NOUN',
                           'in such @WORK_NOUN' 
                           ],

    'THEM_FORMULAIC': [ 'according to CITATION',
                        'like CITATION',
                        'CITATION style',
                        'a la CITATION',
                        'CITATION - style' ] ,

    'US_PREVIOUS_FORMULAIC': [ '@SELF_NOM have previously',
                               '@SELF_NOM have earlier'
                               '@SELF_NOM have elsewhere',
                               '@SELF_NOM elsewhere',
                               '@SELF_NOM previously',
                               '@SELF_NOM earlier',
                               'elsewhere @SELF_NOM',
                               'elswhere @SELF_NOM',
                               'elsewhere , @SELF_NOM',
                               'elswhere , @SELF_NOM',
                               'presented elswhere',
                               'presented elsewhere',
                               '@SELF_NOM have @ARGUMENTATION elsewhere',
                               '@SELF_NOM have @SOLUTION elsewhere',
                               '@SELF_NOM have argue elsewhere',
                               '@SELF_NOM have show elswhere_NOM',
                               '@SELF_NOM have argue elswhere_NOM',
                               '@SELF_NOM will show elsewhere',
                               '@SELF_NOM will show elswhere',
                               '@SELF_NOM will argue elsewhere',
                               '@SELF_NOM will argue elswhere',
                               'elsewhere SELFCITE',
                               'elswhere SELFCITE',
                               'in a @BEFORE_ADJ @PRESENTATION_NOUN',
                               'in an earlier @PRESENTATION_NOUN',
                               'another @PRESENTATION_NOUN' ],
                               
    'TEXTSTRUCTURE_FORMULAIC': [ 'then @SELF_NOM describe',
                                 'then , @SELF_NOM describe',
                                 'next @SELF_NOM describe',
                                 'next , @SELF_NOM describe',
                                 'finally @SELF_NOM describe',
                                 'finally , @SELF_NOM describe',
                                 'then @SELF_NOM present',
                                 'then , @SELF_NOM present',
                                 'next @SELF_NOM present',
                                 'next , @SELF_NOM present',
                                 'finally @SELF_NOM present',
                                 'finally , @SELF_NOM present',
                                 'briefly describe',
                                 'briefly introduce',
                                 'briefly present',
                                 'briefly discuss' ],

    'HERE_FORMULAIC': [ 'in this @PRESENTATION_NOUN',
                        'the present @PRESENTATION_NOUN',
                        '@SELF_NOM here',
                        'here @SELF_NOM',
                        'here , @SELF_NOM',
                        '@GIVEN here',
                        '@SELF_NOM now',
                        'now @SELF_NOM',
                        'now , @SELF_NOM',
                        '@GIVEN now',
                        'herein' ],

    'METHOD_FORMULAIC': [ 'a new @WORK_NOUN',
                          'a novel @WORK_NOUN',
                          'a @WORK_NOUN of',
                          'an @WORK_NOUN of',
                          'a #JJ @WORK_NOUN of',
                          'an #JJ @WORK_NOUN of',
                          'a #NN @WORK_NOUN of',
                          'an #NN @WORK_NOUN of',
                          'a #JJ #NN @WORK_NOUN of',
                          'an #JJ #NN @WORK_NOUN of',
                          'a @WORK_NOUN for',
                          'an @WORK_NOUN for',
                          'a #JJ @WORK_NOUN for',
                          'an #JJ @WORK_NOUN for',
                          'a #NN @WORK_NOUN for',
                          'an #NN @WORK_NOUN for',
                          'a #JJ #NN @WORK_NOUN for',
                          'an #JJ #NN @WORK_NOUN for',
                          '@WORK_NOUN design to #VV', #diff
                          '@WORK_NOUN intend for',
                          '@WORK_NOUN for #VV',
                          '@WORK_NOUN for the #NN',
                          '@WORK_NOUN design to #VV', # diff
                          '@WORK_NOUN to the #NN',
                          '@WORK_NOUN to #NN',
                          '@WORK_NOUN to #VV',
                          '@WORK_NOUN for #JJ #VV', # diff
                          '@WORK_NOUN for the #JJ #NN,'
                          '@WORK_NOUN to the #JJ #NN',
                          '@WORK_NOUN to #JJ #VV',
                          'the problem of #RB #VV',
                          'the problem of #VV', # diff
                          'the problem of how to'], 

    'CONTINUE_FORMULAIC': [ 'follow CITATION',
                            'follow the @WORK_NOUN of CITATION',
                            'follow the @WORK_NOUN give in CITATION',
                            'follow the @WORK_NOUN present in CITATION',
                            'follow the @WORK_NOUN propose in CITATION',
                            'follow the @WORK_NOUN discuss in CITATION',
                            'base on CITATION',
                            '@CONTINUE CITATION',
                            '@CONTINUE the @WORK_NOUN',
                            '@CONTINUE a @WORK_NOUN',
                            '@CONTINUE an @WORK_NOUN',
                            '@CONTINUE @OTHERS_POSS @WORK_NOUN',
                            '@CONTINUE @SELF_POSS @WORK_NOUN',
                            '@AGREE CITATION',
                            '@AGREE the @WORK_NOUN',
                            '@AGREE a @WORK_NOUN',
                            '@AGREE an @WORK_NOUN',
                            '@AGREE @OTHERS_POSS @WORK_NOUN',
                            '@AGREE @SELF_POSS @WORK_NOUN',
                            'base on the @WORK_NOUN of CITATION',
                            'base on the @WORK_NOUN give in CITATION',
                            'base on the @WORK_NOUN present in CITATION',
                            'base on the @WORK_NOUN propose in CITATION',
                            'base on the @WORK_NOUN discuss in CITATION',
                            'adopt CITATION',
                            'start point for @REFERENTIAL @WORK_NOUN',
                            'start point for @SELF_POSS @WORK_NOUN',
                            'as a start point',
                            'as start point',
                            'use CITATION', # dif
                            'base @SELF_POSS',
                            'support @SELF_POSS',
                            'support @OTHERS_POSS',
                            'lend support to @SELF_POSS',
                            'lend support to @OTHERS_POSS',
                            # new
                            '@CONTINUE the @WORK_NOUN of',
                            '@AGREE the @WORK_NOUN of'
                            ],

    'DISCOURSE_CONTRAST_FORMULAIC': [ 'however',
                                      #'nevertheless',
                                      #'nonetheless', 
                                      'unfortunately', 
                                      #'yet',
                                      #'although', 
                                      'whereas' 
                                      ],

    'GRAPHIC_FORMULAIC': [ '@GRAPHIC_NOUN #CD' ],

    'CONTRAST2_FORMULAIC': [ 'this @WORK_NOUN @CONTRAST',
                            '@SELF_POSS @WORK_NOUN @CONTRAST',
                            'this @PRESENTATION_NOUN @CONTRAST',
                            '@SELF_POSS @PRESENTATION_NOUN @CONTRAST',
                            'compare to @OTHERS_POSS @WORK_NOUN',
                            'compare to @OTHERS_POSS @PRESENTATION_NOUN',
                            '@OTHERS_POSS @WORK_NOUN @CONTRAST',
                            'that @WORK_NOUN @CONTRAST',
                            'that @PRESENTATION_NOUN @CONTRAST',
                            '@OTHERS_POSS @PRESENTATION_NOUN @CONTRAST',
                            ],

    'COMPARISON_FORMULAIC': [ 'in @COMPARISON with',
                              'in @COMPARISON to',
                              '@GIVEN #NN @SIMILAR', 
                              '@SELF_POSS #NN @SIMILAR', 
                              '@SELF_POSS @PRESENTATION @SIMILAR', 
                              'a @SELF_POSS @PRESENTATION @SIMILAR', 
                              'a @SIMILAR_ADJ @WORK_NOUN is',
                              'be closely relate to',
                              'be @SIMILAR_ADJ to',
                              'along the line of CITATION',
                              ],
    
    'CONTRAST_FORMULAIC': [ 'against CITATATION',
                              'against @SELF_ACC',
                              'against @SELF_POSS',
                              'against @OTHERS_ACC',
                              'against @OTHERS_POSS',
                              'against @BEFORE_ADJ @WORK_NOUN',
                              'against @MANY @WORK_NOUN',
                              'against @TRADITION_ADJ @WORK_NOUN',
                              'than CITATATION',
                              'than @SELF_ACC',
                              'than @SELF_POSS',
                              'than @OTHERS_ACC',
                              'than @OTHERS_POSS',
                              'than @TRADITION_ADJ @WORK_NOUN',
                              'than @BEFORE_ADJ @WORK_NOUN',
                              'than @MANY @WORK_NOUN',
                              'point of departure from @SELF_POSS',
                              'points of departure from @OTHERS_POSS',
                              'advantage over @OTHERS_ACC',
                              'advantage over @TRADITION_ADJ',
                              'advantage over @MANY @WORK_NOUN',
                              'advantage over @BEFORE_ADJ @WORK_NOUN',
                              'advantage over @OTHERS_POSS',
                              'advantage over CITATATION',
                              'advantage to @OTHERS_ACC',
                              'advantage to @OTHERS_POSS',
                              'advantage to CITATATION',
                              'advantage to @TRADITION_ADJ',
                              'advantage to @MANY @WORK_NOUN',
                              'advantage to @BEFORE_ADJ @WORK_NOUN',
                              'benefit over @OTHERS_ACC',
                              'benefit over @OTHERS_POSS',
                              'benefit over CITATATION',
                              'benefit over @TRADITION_ADJ',
                              'benefit over @MANY @WORK_NOUN',
                              'benefit over @BEFORE_ADJ @WORK_NOUN',
                              'difference to CITATATION',
                              'difference to @TRADITION_ADJ',
                              'difference to CITATATION',
                              'difference to @TRADITION_ADJ',
                              'difference to @MANY @WORK_NOUN',
                              'difference to @BEFORE_ADJ @WORK_NOUN',
                              'difference to @OTHERS_ACC',
                              'difference to @OTHERS_POSS',
                              'difference to @SELF_ACC',
                              'difference to @SELF_POSS',
                              'difference between CITATATION',
                              'difference between @TRADITION_ADJ',
                              'difference between @MANY @WORK_NOUN',
                              'difference between @BEFORE_ADJ @WORK_NOUN',
                              'difference between @OTHERS_ACC',
                              'difference between @OTHERS_POSS',
                              'difference between @SELF_ACC',
                              'difference between @SELF_POSS',
                              'contrast with CITATATION',
                              'contrast with @TRADITION_ADJ',
                              'contrast with @MANY @WORK_NOUN',
                              'contrast with @BEFORE_ADJ @WORK_NOUN',
                              'contrast with @OTHERS_ACC',
                              'contrast with @OTHERS_POSS',
                              'contrast with @SELF_ACC',
                              'contrast with @SELF_POSS',
                              'unlike @SELF_ACC',
                              'unlike @SELF_POSS',
                              'unlike CITATATION',
                              'unlike @TRADITION_ADJ',
                              'unlike @BEFORE_ADJ @WORK_NOUN',
                              'unlike @MANY @WORK_NOUN',
                              'unlike @OTHERS_ACC',
                              'unlike @OTHERS_POSS',
                              'in contrast to @SELF_ACC',
                              'in contrast to @SELF_POSS',
                              'in contrast to CITATATION',
                              'in contrast to @TRADITION_ADJ',
                              'in contrast to @MANY @WORK_NOUN',
                              'in contrast to @BEFORE_ADJ @WORK_NOUN',
                              'in contrast to @OTHERS_ACC',
                              'in contrast to @OTHERS_POSS',
                              'as oppose to @SELF_ACC',
                              'as oppose to @SELF_POSS',
                              'as oppose to CITATATION',
                              'as oppose to @TRADITION_ADJ',
                              'as oppose to @MANY @WORK_NOUN',
                              'as oppose to @BEFORE_ADJ @WORK_NOUN',
                              'as oppose to @OTHERS_ACC',
                              'as oppose to @OTHERS_POSS',
                              'contrary to @SELF_ACC',
                              'contrary to @SELF_POSS',
                              'contrary to CITATATION',
                              'contrary to @TRADITION_ADJ',
                              'contrary to @MANY @WORK_NOUN',
                              'contrary to @BEFORE_ADJ @WORK_NOUN',
                              'contrary to @OTHERS_ACC',
                              'contrary to @OTHERS_POSS',
                              'whereas @SELF_ACC',
                              'whereas @SELF_POSS',
                              'whereas CITATATION',
                              'whereas @TRADITION_ADJ',
                              'whereas @BEFORE_ADJ @WORK_NOUN',
                              'whereas @MANY @WORK_NOUN',
                              'whereas @OTHERS_ACC',
                              'whereas @OTHERS_POSS',
                              'compare to @SELF_ACC',
                              'compare to @SELF_POSS',
                              'compare to CITATATION',
                              #'compare to @TRADITION_ADJ',
                              'compare to @BEFORE_ADJ @WORK_NOUN',
                              'compare to @MANY @WORK_NOUN',
                              'compare to @OTHERS_ACC',
                              'compare to @OTHERS_POSS',
                              'in comparison to @SELF_ACC',
                              'in comparison to @SELF_POSS',
                              'in comparison to CITATATION',
                              'in comparison to @TRADITION_ADJ',
                              'in comparison to @MANY @WORK_NOUN',
                              'in comparison to @BEFORE_ADJ @WORK_NOUN',
                              'in comparison to @OTHERS_ACC',
                              'in comparison to @OTHERS_POSS',
                              'while @SELF_NOM',
                              'while @SELF_POSS',
                              'while CITATATION',
                              'while @TRADITION_ADJ',
                              'while @BEFORE_ADJ @WORK_NOUN',
                              'while @MANY @WORK_NOUN',
                              'while @OTHERS_NOM',
                              'while @OTHERS_POSS',
                              'this @WORK_NOUN @COMPARISON',
                              '@SELF_POSS @WORK_NOUN @COMPARISON',
                              'this @PRESENTATION_NOUN @COMPARISON',
                              '@SELF_POSS @PRESENTATION_NOUN @COMPARISON',
                              'compare to @OTHERS_POSS @WORK_NOUN',
                              'compare to @OTHERS_POSS @PRESENTATION_NOUN',
                              '@OTHERS_POSS @WORK_NOUN @COMPARISON',
                              'that @WORK_NOUN @COMPARISON',
                              'that @PRESENTATION_NOUN @COMPARISON',
                              '@OTHERS_POSS @PRESENTATION_NOUN @COMPARISON',
                              ],
    'ALIGN_FORMULAIC': ['in the @SENSE_NOUN of CITATION'],

    'AFFECT_FORMULAIC': [ 'hopefully', 'thankfully', 'fortunately', 'unfortunately' ],

    'GOOD_FORMULAIC': [ '@GOOD_ADJ' ],
    #'BAD_FORMULAIC': [ '@BAD_ADJ' ],
    'TRADITION_FORMULAIC': [ '@TRADITION_ADJ' ],
    'IN_ORDER_TO_FORMULAIC': [ 'in order to' ],

    'DETAIL_FORMULAIC': ['@SELF_NOM have also',
                         '@SELF_NOM also',
                         'this @PRESENTATION_NOUN also',
                         'this @PRESENTATION_NOUN has also' ],
    
    'NO_TEXTSTRUCTURE_FORMULAIC': [ '( @TEXT_NOUN CREF )',
                                    'as explain in @TEXT_NOUN CREF',
                                    'as explain in the @BEFORE_ADJ @TEXT_NOUN',
                                    'as @GIVEN early in this @TEXT_NOUN',
                                    'as @GIVEN below',
                                    'as @GIVEN in @TEXT_NOUN CREF',
                                    'as @GIVEN in the @BEFORE_ADJ @TEXT_NOUN',
                                    'as @GIVEN in the next @TEXT_NOUN',
                                    '#NN @GIVEN in @TEXT_NOUN CREF',
                                    '#NN @GIVEN in the @BEFORE_ADJ @TEXT_NOUN',
                                    '#NN @GIVEN in the next @TEXT_NOUN',
                                    '#NN @GIVEN below',
                                    'cf. @TEXT_NOUN CREF',
                                    'cf. @TEXT_NOUN below',
                                    'cf. the @TEXT_NOUN below',
                                    'cf. the @BEFORE_ADJ @TEXT_NOUN',
                                    'cf. @TEXT_NOUN above',
                                    'cf. the @TEXT_NOUN above',
                                    'cfXXX @TEXT_NOUN CREF',
                                    'cfXXX @TEXT_NOUN below',
                                    'cfXXX the @TEXT_NOUN below',
                                    'cfXXX the @BEFORE_ADJ @TEXT_NOUN',
                                    'cfXXX @TEXT_NOUN above',
                                    'cfXXX the @TEXT_NOUN above',
                                    'e. g. , @TEXT_NOUN CREF',
                                    'e. g , @TEXT_NOUN CREF',
                                    'e. g. @TEXT_NOUN CREF',
                                    'e. g @TEXT_NOUN CREF',
                                    'e.g., @TEXT_NOUN CREF',
                                    'e.g. @TEXT_NOUN CREF',
                                    'compare @TEXT_NOUN CREF',
                                    'compare @TEXT_NOUN below',
                                    'compare the @TEXT_NOUN below',
                                    'compare the @BEFORE_ADJ @TEXT_NOUN', 
                                    'compare @TEXT_NOUN above',
                                    'compare the @TEXT_NOUN above',
                                    'see @TEXT_NOUN CREF',
                                    'see the @BEFORE_ADJ @TEXT_NOUN',
                                    'recall from the @BEFORE_ADJ @TEXT_NOUN',
                                    'recall from the @TEXT_NOUN above',
                                    'recall from @TEXT_NOUN CREF',
                                    '@SELF_NOM shall see below',
                                    '@SELF_NOM will see below',
                                    '@SELF_NOM shall see in the next @TEXT_NOUN',
                                    '@SELF_NOM will see in the next @TEXT_NOUN',
                                    '@SELF_NOM shall see in @TEXT_NOUN CREF',
                                    '@SELF_NOM will see in @TEXT_NOUN CREF',
                                    'example in @TEXT_NOUN CREF',
                                    'example CREF in @TEXT_NOUN CREF',
                                    'example CREF and CREF in @TEXT_NOUN CREF',
                                    'example in @TEXT_NOUN CREF' ],

    'USE_FORMULAIC': [ '@SELF_NOM @USE',
                       #'@WORK_NOUN @USE',
                       '@SELF_NOM @RESEARCH',
                       #'be @USE to',
                       #'can be #VV use', #can be /solved/ using
                       #'@SELF_POSS @WORK_NOUN be @CONTINUE',
                       #'@SELF_POSS #JJ @WORK_NOUN be @CONTINUE',
                       '@SOLUTION with the @HELP_NOUN of',
                       '@SOLUTION with the @WORK_NOUN of',
                       ],

    'FUTURE_WORK_FORMULAIC': [ '@FUTURE_ADJ @WORK_NOUN',
                               '@FUTURE_ADJ @AIM_NOUN',
                               '@FUTURE_ADJ @CHANGE_NOUN',
                               'a @HEDGE_ADJ @AIM_NOUN',
                               'one @HEDGE_ADJ @AIM_NOUN',
                               '#NN be also @HEDGE_ADJ',
                               'in the future',
                               '@SELF_NOM @FUTURE_INTEREST',
                               ],

    'HEDGING_FORMULAIC': [ '@HEDGING_MODALS be @RESEARCH',
                           '@HEDGING_MODALS be @CHANGE',
                           '@HEDGING_MODALS be @SOLUTION',
                           ],

    'PRESENT_WORK_FORMULAIC': [ '@SELF_NOM be @CURRENT_ADV @RESEARCH',
                                '@SELF_NOM be @RESEARCH @CURRENT_ADV'],

    'EXTENDING_WORK_FORMULAIC': [ '@CHANGE the @WORK_NOUN',
                                  '@CHANGE this @WORK_NOUN',
                                  '@SELF_POSS @WORK_NOUN be @CHANGE',
                                  '@SELF_POSS #JJ @WORK_NOUN be @CHANGE',
                                  '@SELF_POSS @WORK_NOUN @CHANGE',
                                  '@SELF_POSS #JJ @WORK_NOUN @CHANGE',
                                  '@CHANGE the #JJ @WORK_NOUN',
                                  '@SELF_NOM @CHANGE'
                                  ],

    'EXTENDING_WORK2_FORMULAIC': [ '@SELF_NOM @CHANGE #DD @WORK_NOUN',
                                   '@SELF_POSS @WORK_NOUN @CHANGE',
                                   '@CHANGE from CITATION',
                                   '@CHANGE from #NN of CITATION',
                                   '@SELF_POSS @CHANGE_NOUN of CITATION',
                                   '@SELF_POSS @WORK_NOUN @CONTINUE',
                                   '@SELF_POSS @WORK_NOUN be #DD @CHANGE_NOUN',
                                   '@SELF_POSS @WORK_NOUN be #VV #DD @CHANGE_NOUN',
                                   '#NN be #DD @CHANGE_NOUN of',
                                   '#NN be #DD #JJ @CHANGE_NOUN of',
                                   '#DD #NN @DENOTATION #DD @CHANGE_NOUN of',
                                   '@TEXT_NOUN @CONTINUE CITATION',
                                   '#NN @CONTINUE #NN of CITATION',
                                   'be @SEE as an @CHANGE_NOUN',
                                   '@CHANGE #DD #NN of CITATION',
                                  ],

    'USEFUL_FORMULAIC': [ 'have shown @GOOD_ADJ for' ],

    'MOTIVATING_FORMULAIC': [ 'as @PRESENTATION in CITATION',
                              'as @PRESENTATION by CITATION',
                              'this be a #JJ convention',
                              'this be a #RB #JJ convention',
                              '@CONTINUE the #NN result',
                              '@CONTINUE the #JJ result',
                              '@AGREE the #NN result',
                              '@AGREE the #JJ result',

                              #'@INSPRATION by the #NN result',
                              #'@INSPIRATION by the #JJ result',
                              '@INSPIRATION by',
                              'CITATION have @PRESENTATION that',
                              'have remain a @PROBLEM_NOUN',

                              'their importance have @INCREASE',

                              '#NN be @MAIN_ADJ in',
                              '#NN be @MAIN_ADJ for',
                              
                              'it be @MAIN_ADJ not to',

                              'from CITATION , @SELF_NOM',
                              
                              '@CONTINUE CITATION, @SELF_NOM',
                              '@AGREE CITATION, @SELF_NOM',
                              '@RESEARCH in @DISCIPLINE @PRESENTATION',
                              '@RESEARCH in #NN @PRESENTATION',
                              '@RESEARCH in #NN #NN @PRESENTATION',
                              '@RESEARCH in #JJ #NN @PRESENTATION',

                              'negative @RESULT_NOUN for',
                              'negative @RESULT_NOUN that',
                              
                              # 'have be @PRESENTATION',  # SUPER NOISY :( 
                              'it be well document',
                              'it have be well document',
                              '#NN need to @USE',

                              'CITATION have @RESEARCH it',
                              'CITATION have @PRESENTATION that',
                              'CITATATION @PRESENTATION that',
                              'CITATATION #RB @PRESENTATION that',

                              'prove to be @GOOD_ADJ in',
                              '@PRESENTATION to be @GOOD_ADJ in',
                              'prove to be @GOOD_ADJ for',
                              '@PRESENTATION to be @GOOD_ADJ for',
                              ],

    "PRIOR_WORK_FORMULAIC": [ '@BEFORE_ADJ @PRESENTATION @SELF_NOM',
                              '@BEFORE_ADJ @PRESENTATION , @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION , @SELF_NOM',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION @SELF_NOM',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION , @SELF_NOM',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION CITATION',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION SELFCITATION',
                              '@BEFORE_ADJ @PRESENTATION CITATION @SELF_NOM',
                              '@BEFORE_ADJ @PRESENTATION CITATION , @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION CITATION @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION CITATION , @SELF_NOM',
                              'first @PRESENTATION in CITATION',
                              '@PRESENTATION #RR in CITATION',
                              '@PRESENTATION #JJ in CITATION',
                              '@BEFORE_ADJ @CHANGE_NOUN of @SELF_POSS @WORK_NOUN',
                              '@CHANGE on @BEFORE_ADJ @PRESENTATION @PRESENTATION in SELFCITATION',
                              '@CHANGE @BEFORE_ADJ @PRESENTATION @PRESENTATION in SELFCITATION',
                              '@CHANGE @BEFORE_ADJ @PRESENTATION @PRESENTATION in SELFCITATION',
                              '@CHANGE @BEFORE_ADJ @PRESENTATION @PRESENTATION SELFCITATION',
                              '@CHANGE on @SELF_POSS @BEFORE_ADJ @PRESENTATION @PRESENTATION in SELFCITATION',
                              '@CHANGE @SELF_POSS @BEFORE_ADJ @PRESENTATION @PRESENTATION in SELFCITATION',
                              '@CHANGE @SELF_POSS @BEFORE_ADJ @PRESENTATION @PRESENTATION in SELFCITATION',
                              '@CHANGE @SELF_POSS @BEFORE_ADJ @PRESENTATION @PRESENTATION SELFCITATION',
                              'in @SELF_POSS @BEFORE_ADJ @PRESENTATION CITATION',
                              ]        
    }
                              


AGENT_PATTERNS = {

    'US_AGENT': [ '@SELF_NOM',
                  '@SELF_POSS #JJ @WORK_NOUN',
                  '@SELF_POSS #JJ @PRESENTATION_NOUN',
                  '@SELF_POSS #JJ @ARGUMENTATION_NOUN',
                  '@SELF_POSS #JJ @SOLUTION_NOUN',
                  '@SELF_POSS #JJ @RESULT_NOUN',
                  '@SELF_POSS @WORK_NOUN',
                  '@SELF_POSS @PRESENTATION_NOUN',
                  '@SELF_POSS @ARGUMENTATION_NOUN',
                  '@SELF_POSS @SOLUTION_NOUN',
                  'SELF_POSS @RESULT_NOUN',
                  '@WORK_NOUN @GIVEN here',
                  'WORK_NOUN @GIVEN below',
                  '@WORK_NOUN @GIVEN in this @PRESENTATION_NOUN',
                  '@WORK_NOUN @GIVEN in @SELF_POSS @PRESENTATION_NOUN',
                  'the @SOLUTION_NOUN @GIVEN here',
                  'the @SOLUTION_NOUN @GIVEN in this @PRESENTATION_NOUN',
                  'the first author',
                  'the second author',
                  'the third author',
                  'one of the authors',
                  'one of us' ],

    'REF_US_AGENT': [ 'this @PRESENTATION_NOUN',
                      'the present @PRESENTATION_NOUN',
                      'the current @PRESENTATION_NOUN',
                      'the present #JJ @PRESENTATION_NOUN',
                      'the current #JJ @PRESENTATION_NOUN',
                      'the @WORK_NOUN @GIVEN' ],

    'OUR_AIM_AGENT': [ '@SELF_POSS @AIM_NOUN',
                       'the point of this @PRESENTATION_NOUN',
                       'the @AIM_NOUN of this @PRESENTATION_NOUN',
                       'the @AIM_NOUN of the @GIVEN @WORK_NOUN',
                       'the @AIM_NOUN of @SELF_POSS @WORK_NOUN',
                       'the @AIM_NOUN of @SELF_POSS @PRESENTATION_NOUN',
                       'the most @MAIN_ADJ feature of @SELF_POSS @WORK_NOUN',
                       'contribution of this @PRESENTATION_NOUN',
                       'contribution of the @GIVEN @WORK_NOUN',
                       'contribution of @SELF_POSS @WORK_NOUN',
                       'the question @GIVEN in this PRESENTATION_NOUN',
                       'the question @GIVEN here',
                       '@SELF_POSS @MAIN_ADJ @AIM_NOUN',
                       '@SELF_POSS @AIM_NOUN in this @PRESENTATION_NOUN',
                       '@SELF_POSS @AIM_NOUN here',
                       'the #JJ point of this @PRESENTATION_NOUN',
                       'the #JJ purpose of this @PRESENTATION_NOUN',
                       'the #JJ @AIM_NOUN of this @PRESENTATION_NOUN',
                       'the #JJ @AIM_NOUN of the @GIVEN @WORK_NOUN',
                       'the #JJ @AIM_NOUN of @SELF_POSS @WORK_NOUN',
                       'the #JJ @AIM_NOUN of @SELF_POSS @PRESENTATION_NOUN',
                       'the #JJ question @GIVEN in this PRESENTATION_NOUN',
                       'the #JJ question @GIVEN here' ],

    'AIM_REF_AGENT':  [ 'its @AIM_NOUN',
                        'its #JJ @AIM_NOUN',
                        '@REFERENTIAL #JJ @AIM_NOUN',
                        'contribution of this @WORK_NOUN',
                        'the most important feature of this @WORK_NOUN',
                        'feature of this @WORK_NOUN',
                        'the @AIM_NOUN',
                        'the #JJ @AIM_NOUN' ],
                        
    'US_PREVIOUS_AGENT': [ 'SELFCITATION',
                           'this @BEFORE_ADJ @PRESENTATION_NOUN',
                           '@SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN',
                           '@SELF_POSS @BEFORE_ADJ @WORK_NOUN',
                           'in CITATION , @SELF_NOM',
                           'in CITATION @SELF_NOM',
                           'the @WORK_NOUN @GIVEN in SELFCITATION',
                           'in @BEFORE_ADJ @PRESENTATION CITATION @SELF_NOM',
                           'in @BEFORE_ADJ @PRESENTATION CITATION , @SELF_NOM',
                           'in a @BEFORE_ADJ @PRESENTATION CITATION @SELF_NOM',
                           'in a @BEFORE_ADJ @PRESENTATION CITATION , @SELF_NOM',
                           ],

    'REF_AGENT': [ '@REFERENTIAL #JJ @WORK_NOUN',
                   #'@REFERENTIAL @WORK_NOUN',
                   'this sort of @WORK_NOUN',
                   'this kind of @WORK_NOUN',
                   'this type of @WORK_NOUN',
                   'the current #JJ @WORK_NOUN',
                   'the current @WORK_NOUN',
                   'the @WORK_NOUN',
                   'the @PRESENTATION_NOUN',
                   'the author',
                   'the authors' ],

    'THEM_PRONOUN_AGENT': [ '@OTHERS_NOM' ],    

    'THEM_ACTIVE_AGENT' : [ 'CITATION @PRESENTATION' ] ,

    'THEM_AGENT': [ 'CITATION',
                    'CITATION \'s #NN',
                    'CITATION \'s @PRESENTATION_NOUN',
                    'CITATION \'s @WORK_NOUN',
                    'CITATION \'s @ARGUMENTATION_NOUN',
                    'CITATION \'s #JJ @PRESENTATION_NOUN',
                    'CITATION \'s #JJ @WORK_NOUN',
                    'CITATION \'s #JJ @ARGUMENTATION_NOUN',
                    'the CITATION @WORK_NOUN',
                    'the @WORK_NOUN @GIVEN in CITATION',
                    'the @WORK_NOUN of CITATION',
                    '@OTHERS_POSS @PRESENTATION_NOUN',
                    '@OTHERS_POSS @WORK_NOUN',
                    '@OTHERS_POSS @RESULT_NOUN',
                    '@OTHERS_POSS @ARGUMENTATION_NOUN',
                    '@OTHERS_POSS @SOLUTION_NOUN',
                    '@OTHERS_POSS #JJ @PRESENTATION_NOUN',
                    '@OTHERS_POSS #JJ @WORK_NOUN',
                    '@OTHERS_POSS #JJ @RESULT_NOUN',
                    '@OTHERS_POSS #JJ @ARGUMENTATION_NOUN',
                    '@OTHERS_POSS #JJ @SOLUTION_NOUN' ],

    'GAP_AGENT':  [ 'none of these @WORK_NOUN',
                    'none of those @WORK_NOUN',
                    'no @WORK_NOUN',
                    'no #JJ @WORK_NOUN',
                    'none of these @PRESENTATION_NOUN',
                    'none of those @PRESENTATION_NOUN',
                    'no @PRESENTATION_NOUN',
                    'no #JJ @PRESENTATION_NOUN' ],

    'GENERAL_AGENT': [ '@TRADITION_ADJ #JJ @WORK_NOUN',
                       '@TRADITION_ADJ use @WORK_NOUN',
                       '@TRADITION_ADJ @WORK_NOUN',
                       '@MANY #JJ @WORK_NOUN',
                       '@MANY @WORK_NOUN',
                       '@BEFORE_ADJ #JJ @WORK_NOUN',
                       '@BEFORE_ADJ @WORK_NOUN',
                       '@BEFORE_ADJ #JJ @PRESENTATION_NOUN',
                       '@BEFORE_ADJ @PRESENTATION_NOUN',
                       'other #JJ @WORK_NOUN',
                       'other @WORK_NOUN',
                       'such @WORK_NOUN',
                       'these #JJ @PRESENTATION_NOUN',
                       'these @PRESENTATION_NOUN',
                       'those #JJ @PRESENTATION_NOUN',
                       'those @PRESENTATION_NOUN',
                       '@REFERENTIAL authors',
                       '@MANY author',
                       'researcher in @DISCIPLINE',
                       '@PROFESSIONALS' ],

    'PROBLEM_AGENT': [ '@REFERENTIAL #JJ @PROBLEM_NOUN',
                       '@REFERENTIAL @PROBLEM_NOUN',
                       'the @PROBLEM_NOUN' ],

    'SOLUTION_AGENT': [ '@REFERENTIAL #JJ @SOLUTION_NOUN',
                       '@REFERENTIAL @SOLUTION_NOUN',
                       'the @SOLUTION_NOUN',
                       'the #JJ @SOLUTION_NOUN' ],

    'TEXTSTRUCTURE_AGENT': [ '@TEXT_NOUN CREF',
                             '@TEXT_NOUN CREF and CREF',
                             'this @TEXT_NOUN',
                             'next @TEXT_NOUN',
                             'next #CD @TEXT_NOUN',
                             'concluding @TEXT_NOUN',
                             '@BEFORE_ADJ @TEXT_NOUN',
                             '@TEXT_NOUN above',
                             '@TEXT_NOUN below',
                             'following @TEXT_NOUN',
                             'remaining @TEXT_NOUN',
                             'subsequent @TEXT_NOUN',
                             'following #CD @TEXT_NOUN',
                             'remaining #CD @TEXT_NOUN',
                             'subsequent #CD @TEXT_NOUN',
                             '@TEXT_NOUN that follow',
                             'rest of this @PRESENTATION_NOUN',
                             'remainder of this @PRESENTATION_NOUN',
                             'in @TEXT_NOUN CREF , @SELF_NOM',
                             'in this @TEXT_NOUN , @SELF_NOM',
                             'in the next @TEXT_NOUN , @SELF_NOM',
                             'in @BEFORE_ADJ @TEXT_NOUN , @SELF_NOM',
                             'in the @BEFORE_ADJ @TEXT_NOUN , @SELF_NOM',
                             'in the @TEXT_NOUN above , @SELF_NOM',
                             'in the @TEXT_NOUN below , @SELF_NOM',
                             'in the following @TEXT_NOUN , @SELF_NOM',
                             'in the remaining @TEXT_NOUN , @SELF_NOM',
                             'in the subsequent @TEXT_NOUN , @SELF_NOM',
                             'in the @TEXT_NOUN that follow , @SELF_NOM',
                             'in the rest of this @PRESENTATION_NOUN , @SELF_NOM',
                             'in the remainder of this @PRESENTATION_NOUN , @SELF_NOM',
                             'below , @SELF_NOM',
                             'the @AIM_NOUN of this @TEXT_NOUN' ]
    }


    
# Split the patterns into lists of things to match
for feature_name in list(FORMULAIC_PATTERNS.keys()):
    tokenized = []
    patterns = FORMULAIC_PATTERNS[feature_name]
    for pattern in patterns:
        tokenized.append(pattern.split())
    FORMULAIC_PATTERNS[feature_name] = tokenized

for feature_name in list(AGENT_PATTERNS.keys()):
    tokenized = []
    patterns = AGENT_PATTERNS[feature_name]
    for pattern in patterns:
        tokenized.append(pattern.split())
    AGENT_PATTERNS[feature_name] = tokenized
        
def get_custom_pattern_features(citance, pre_sents, post_sents, cite_index, debug=None, prefix=None):
    features = Counter() # {}
    global CUR_LABEL
    global FEATURE_FIRING_COUNTS

    #
    # Get the features for the citance
    #
    for (feature_name, patterns) in CUSTOM_IN_CITANCE_PATTERNS.iteritems():
        if prefix is not None:
            feature_name = prefix + feature_name
        for pattern in patterns:
            pat_index = find(pattern, citance, None, debug=debug, feature=feature_name)
            if pat_index < 0:
                continue
            if debug is not None:
                print 'found %s in %s' % (pattern, debug)

            # If the pattern happens after the citation
            if cite_index < pat_index:
                offset = pat_index - cite_index
            # Otherwise, it happens before, so take into account its length
            else:
                offset = (pat_index-len(pattern)) - cite_index                

            features[feature_name] += 1
            if CUR_LABEL is not None:
                FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name, CUR_LABEL.split("-")[0])] += 1

    for pre_sent in pre_sents:
        for (feature_name, patterns) in CUSTOM_PRE_CITANCE_PATTERNS.iteritems():
            if prefix is not None:
                feature_name = prefix + feature_name
            for pattern in patterns:
                pat_index = find(pattern, pre_sent, None, debug=debug, feature=feature_name)
                if pat_index < 0:
                    continue
                if debug is not None:
                    print 'found %s in %s' % (pattern, debug)

                # If the pattern happens after the citation
                if cite_index < pat_index:
                    offset = pat_index - cite_index
            # Otherwise, it happens before, so take into account its length
                else:
                    offset = (pat_index-len(pattern)) - cite_index                

                features[feature_name] += 1
                if CUR_LABEL is not None:
                    FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name, CUR_LABEL.split("-")[0])] += 1
                    

    for post_sent in post_sents:
        for (feature_name, patterns) in CUSTOM_POST_CITANCE_PATTERNS.iteritems():
            if prefix is not None:
                feature_name = prefix + feature_name
            for pattern in patterns:
                pat_index = find(pattern, post_sent, None, debug=debug, feature=feature_name)
                if pat_index < 0:
                    continue
                if debug is not None:
                    print 'found %s in %s' % (pattern, debug)

                # If the pattern happens after the citation
                if cite_index < pat_index:
                    offset = pat_index - cite_index
            # Otherwise, it happens before, so take into account its length
                else:
                    offset = (pat_index-len(pattern)) - cite_index                

                features[feature_name] += 1
                if CUR_LABEL is not None:
                    FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name, CUR_LABEL.split("-")[0])] += 1




    return features

def get_formulaic_features(processed_sentence, cite_index, debug=None, prefix=None):
    features = {}
    global CUR_LABEL
    global FEATURE_FIRING_COUNTS



    for (feature_name, patterns) in FORMULAIC_PATTERNS.iteritems():
        if prefix is not None:
            feature_name = prefix + feature_name

        
        #if feature_name == 'EXTENDING_WORK2_FORMULAIC':
        #    print "testing for", feature_name
        #    print to_str(processed_sentence)
        #    debug=True
        #else:
        #    debug=None

        for pattern in patterns:
            pat_index = find(pattern, processed_sentence, None, debug=debug, feature=feature_name)
            if pat_index >= 0:
                if debug is not None:
                    print 'found %s in %s' % (pattern, debug)

                # If the pattern happens after the citation
                if cite_index < pat_index:
                    offset = pat_index - cite_index
                # Otherwise, it happens before, so take into account its length
                else:
                    offset = (pat_index-len(pattern)) - cite_index                

                features[feature_name] = offset
                if CUR_LABEL is not None:
                    FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name, CUR_LABEL.split("-")[0])] += 1
                break


    # Additionally, the 168 agent patterns are also considered as formulaic
    # patterns, wherever they do not occur as the subject of the sentence. The
    # decision to include these into the Formu feature was explained in section
    # 5.2.2.2.
    for (feature_name, patterns) in AGENT_PATTERNS.iteritems():
        if prefix is not None:
            feature_name = prefix + feature_name

        for pattern in patterns:
            pat_index = find(pattern, processed_sentence, False,  debug=debug, feature=feature_name + " (FORMULAIC)")
            if pat_index >= 0:
                if debug is not None:
                    print 'found %s (as FORMULAIC) in %s' % (pattern, debug)

                # If the pattern happens after the citation
                if cite_index < pat_index:
                    offset = pat_index - cite_index
                # Otherwise, it happens before, so take into account its length
                else:
                    offset = (pat_index-len(pattern)) - cite_index                

                features[feature_name + "_AS_FORMULAIC"] = offset
                if CUR_LABEL is not None:
                    FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name + ' (AS_FORM)', CUR_LABEL.split("-")[0])] += 1

                break


    return features


def get_agent_features(processed_sentence, cite_index, debug=None, prefix=None):
    features = {}
    global CUR_LABEL
    global FEATURE_FIRING_COUNTS

    for (feature_name, patterns) in AGENT_PATTERNS.iteritems():
        if prefix is not None:
            feature_name = prefix + feature_name
        
        for pattern in patterns:
            pat_index = find(pattern, processed_sentence, True, debug=debug)
            if pat_index >= 0:
                if debug is not None:
                    print 'found %s in %s' % (pattern, debug)

                # If the pattern happens after the citation
                if cite_index < pat_index:
                    offset = pat_index - cite_index
                # Otherwise, it happens before, so take into account its length
                else:
                    offset = (pat_index-len(pattern)) - cite_index
                    
                features[feature_name] = offset
                if CUR_LABEL is not None:
                    FEATURE_FIRING_COUNTS[(' '.join(pattern), feature_name + ' (AS_AGENT)', CUR_LABEL.split("-")[0])] += 1

                break


    return features



def find(pattern, sentence, must_have_subj_value, debug=None, feature=None):
    pat_len = len(pattern)

    #if debug is not None:
    #    print json.dumps(sentence)

    if debug is not None:
        print '\n\ntesting %s (%s) against "%s" (must be subj? %s)' % (pattern, feature, debug, must_have_subj_value)



    # For each position in the sentence
    for i in range(0, (len(sentence) - pat_len) + 1):
        
        match = True
        is_subj = False        
        # This is the adjustment to the sentence's token offset based on finding
        # a MWE match in a lexicon
        k = 0 


        if debug is not None:
            print 'starting search at ' + sentence[i]['word']

        for j in range(0, pat_len):

            if debug is not None:
                print '%d:%d:%d -> "%s" in "%s"?' % (i, j, k, sentence[i+j+k]['lemma'], pattern[j])

            # Check that we won't search outside the sentence length due to
            # finding a MWE lexicon entry at the end of the sentence
            if i+j+k >= len(sentence):
                if debug is not None:
                    print 'moved beyond end of sentence :('
                match = False
                break

            # print '%d %s' % (i+j+k, sentence[i+j+k]['ArgType'])
            if sentence[i+j+k]['ArgType'] == 'subj':
                is_subj = True;

            pi =  pattern[j]
            if debug is not None:
                print 'Testing %d/%d: %s' % (j+1, pat_len, pi)

            # If this is a category that we have to look up
            if pi[0] == '@':
                label = pi[1:]
                if debug is not None:
                    print 'Checking if "%s" is in %s' % (sentence[i+j+k]['lemma'], label)
                lexicon = None
                required_pos = None
                if label in ALL_CONCEPT_LEXICONS:
                    lexicon = ALL_CONCEPT_LEXICONS[label]
                elif label in ALL_ACTION_LEXICONS:
                    lexicon = ALL_ACTION_LEXICONS[label]
                    required_pos = 'V'

                if lexicon is None:
                    raise BaseException(("unknown lexicon ref: '%s' in %s, %s" % (label, feature, pattern)))
                
                (is_match, matched_phrased_length) = is_in_lexicon(lexicon, sentence, i+j+k, required_pos=required_pos)

                #print 'found %s (%d) in %s? %s (%d)' % (sentence[i+j+k]['lemma'], i+j+k, label, is_match, matched_phrased_length)

                if not is_match:
                    match = False
                    break            
                else:
                    if debug is not None:
                        print 'YAY:: "%s" is in set %s in %s' % (sentence[i+j+k]['lemma'], label, debug)

                # If we did find a match, recognize that some phrases are
                # multi-word expressions, so we may need to skip ahead more than
                # one token.  Note that we were already going to skip one token
                # anyway, so substract 1 from the phrase length
                k += (matched_phrased_length - 1)
                    
                #if not sentence[i+j+k]['lemma'] not in lexicon:
                #    if debug is not None:
                #        print '"%s" is not in set %s in %s' % (sentence[i+j+k]['lemma'], label, debug)
                #    match = False
                #    break
                #else:
                #    if debug is not None:
                
            elif pi == 'SELFCITATION':
                if debug is not None:
                    print 'Checking if "%s" is %s' % (sentence[i+j+k]['pos'][0], pi[1])

                if sentence[i+j+k]['word'] != pi:
                    if debug is not None:
                        print '"%s" is not a %s in %s' % (sentence[i+j+k]['lemma'], pi, debug)
                    match = False
                    break
                else:
                    if debug is not None:
                        print 'YAY:: "%s" is a %s in %s' % (sentence[i+j+k]['lemma'], pi, debug)

            elif pi == 'CITATION':
                if debug is not None:
                    print 'Checking if "%s" is %s' % (sentence[i+j+k]['pos'][0], pi[1])

                if not sentence[i+j+k]['word'].endswith(pi):
                    if debug is not None:
                        print '"%s" is not a %s in %s' % (sentence[i+j+k]['lemma'], pi, debug)
                    match = False
                    break
                else:
                    if debug is not None:
                        print 'YAY:: "%s" is a %s in %s' % (sentence[i+j+k]['lemma'], pi, debug)

    
            # Not sure if this is entirely right...
            elif pi == 'CREF':
                if sentence[i+j+k]['pos'] != 'CD' or sentence[i+j+k]['word'] != 'CREF':
                    match = False
                    break
                
            # If this is POS-match
            elif pi[0] == '#':
                if debug is not None:
                    print 'Checking if "%s" is  %s' % (sentence[i+j+k]['pos'][0], pi[1])
                # NOTE: we compare only the coarsest POS tag level (N/V/J)
                #
                # NOTE Check for weird POS-tagging issues with verbal adjectives
                if sentence[i+j+k]['pos'][0] != pi[1] and not (pi[1] == 'J' and sentence[i+j+k]['pos'] == 'VBN'):
                    match = False
                    if debug is not None:
                        print '"%s" is not %s in %s' % (sentence[i+j+k]['pos'][0], pi[1], debug)
                        break
                else:
                    if debug is not None:
                        print '"YAY:: %s" is %s in %s' % (sentence[i+j+k]['pos'][0], pi[1], debug)

            # Otherwise, we have to match the word
            else:
                if debug is not None:
                    print 'Checking if "%s" is %s' % (sentence[i+j+k]['lemma'], pi)
                if sentence[i+j+k]['lemma'] != pi:
                    if debug is not None:
                        print '"%s" is not %s in %s' % (sentence[i+j+k]['lemma'], pi, debug)
                    match = False
                    break
                else:
                    if debug is not None:
                        print 'YAY:: "%s" is %s in %s' % (sentence[i+j+k]['lemma'], pi, debug)       

        if match and (must_have_subj_value is not None) and (is_subj is not must_have_subj_value):
            if debug is not None:
                print 'needed a subject for %s but this isn\'t one (%s != %s)' % (feature, is_subj, must_have_subj_value)
            continue


        # TODO: confirm we can skip 'j' items so i += j
        if match:
            if debug is not None:
                print 'match!\n\n'
            return i
        else:
            if debug is not None:
                print 'no match (%d, %d, %d)\n\n' % (i, j, k)

    if debug is not None:
        print '\n\n'

    return -1

def get_num_sentences(xmldoc):
    # text_sents is a list of all sentence in paper 
    text_sents = getPaperSentences_2(xmldoc)
    return len(text_sents)

#
# The input is one citation in one paper.  If the paper is cited multiple times,
# parcit_context is the context for one citation in one paper.
#
#
def get_citation_sentence_index(parscit_context, xmldoc):

    # text_sents is a list of all sentence in paper 
    text_sents = getPaperSentences_2(xmldoc)

    # this is the citation itself, e.g., "Foo (2000)"
    citingString = parscit_context.attributes['citStr'].value.encode('utf-8')
    citingString = normalize_string(citingString)
    citingString = citingString.strip()
    
    citeIndex = parscit_context.firstChild.data.index(citingString)

    # This is the 9 characters before the citing string
    citationArea = parscit_context.firstChild.data[citeIndex - 10: citeIndex-1].strip()
    citationArea = normalize_string(citationArea)
    citationArea = citationArea.strip().encode('utf-8')
    
    idx = -1
     
    # This loop finds the sentence contains the context for the current citation
    for sent in text_sents:
        idx += 1

        # Look for the citing string, which could potentially match multiple contexts for the same citation
        if sent.find(citingString) == -1:
            continue

        # Check for the citationArea string (the context for this particular
        # citation) to ensure we have the right citation area       
        if ((sent.startswith(citingString) == 1) or (sent.startswith("(" + citingString) == 1)) and text_sents[idx-1].find(citationArea) == -1:
            continue
        elif ((sent.startswith(citingString) == 0) and (not sent.startswith("(" + citingString) == 1)) and sent.find(citationArea) == -1:
            continue

        return idx

    return -1
