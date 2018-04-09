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

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet 

#from global_functions_march16_teufel import *
from global_functions_march16 import *

from scipy import spatial
import numpy as np
import random

PAPER_TO_NUM_CITES = defaultdict(Counter)
PAPER_TO_PAGE_RANK = defaultdict(Counter)
PAPER_TO_HUB = defaultdict(Counter)
PAPER_TO_AUTHORITY = defaultdict(Counter)
PAPER_TO_LOAD_CENTRALITY = defaultdict(Counter)
#with open('../resources/arc-paper-to-temporal-weight.2.tsv') as fh:
with open('../working-files/arc-network-weights.tsv') as f:

    for line_no, line in enumerate(f):
        # Skip header

        if line_no == 0:
            continue
        cols = line[:-1].split('\t')
        year = int(cols[0])
        paper_id = cols[1]
        if not paper_id in PAPER_TO_PAGE_RANK:
            PAPER_TO_PAGE_RANK[paper_id] = defaultdict(Counter)
            PAPER_TO_NUM_CITES[paper_id] = defaultdict(Counter)
            PAPER_TO_HUB[paper_id] = defaultdict(Counter)
            PAPER_TO_AUTHORITY[paper_id] = defaultdict(Counter)
            PAPER_TO_LOAD_CENTRALITY[paper_id] = defaultdict(Counter)
            

        PAPER_TO_PAGE_RANK[paper_id][year] = float(cols[2])
        PAPER_TO_NUM_CITES[paper_id][year] = int(cols[3])
        PAPER_TO_HUB[paper_id][year] = float(cols[4])
        PAPER_TO_AUTHORITY[paper_id][year] = float(cols[5])
        PAPER_TO_LOAD_CENTRALITY[paper_id][year] = float(cols[6])

WORD_TO_VEC = {} # defaultdict(lambda: [ random.random(), random.random() ])

if True:
    # Use a pre-processed subset that contains only those words in the ARC
    with open('../resources/glove.840B.300d.ARC-subset.txt') as f:
        print 'Loading GloVe vectors'
        line_no = 0
        for line in f:
            # break
            cols = line.split()
            WORD_TO_VEC[cols[0]] = np.array(map(float, cols[1:]))
            line_no += 1
            if line_no % 100000 == 0:
                print 'Loaded %d so far...' % (line_no)
                # break
        print 'Done loading GloVe vectors'

def get_citance(citation_context, parsed_doc):
    return parsed_doc['sections'][citation_context['section']]\
        ['subsections'][citation_context['subsection']]\
        ['sentences'][citation_context['sentence']],

def get_subsection(citation_context, parsed_doc):
    return parsed_doc['sections'][citation_context['section']]\
        ['subsections'][citation_context['subsection']]

#
# The input is one citation in one paper.  If the paper is cited multiple times,
# parcit_context is the context for one citation in one paper.
#
def get_context_features(citation_context, parsed_doc):    
    
    citance = get_citance(citation_context, parsed_doc)[0]
    sent = citance['text']
    citing_string = citation_context['citing_string']
    subsection = get_subsection(citation_context, parsed_doc)
    section = parsed_doc['sections'][citation_context['section']]
    sent_index = citation_context['sentence']
    if 'cited_paper_id' in citation_context:
        cited_paper_id = citation_context['cited_paper_id']
    else:
        cited_paper_id = ''
        # print json.dumps(citation_context)
        return {}

    ret_val = OrderedDict()

    CITATION = 'CITATION'
    if citation_context['is_self_cite']:
        CITATION = 'SELFCITATION'

    IS_USED_AS_TEXT = 0
    if '(' in citing_string or ')' in citing_string:
        IS_USED_AS_TEXT = 1

    # Global information features
    citing_year = parsed_doc['year']
    cited_year = int(citation_context['info']['year'])
    ret_val['year_diff'] = citing_year - cited_year
    ret_val['num_cites'] = PAPER_TO_NUM_CITES[cited_paper_id][year]
    ret_val['page_rank'] = PAPER_TO_PAGE_RANK[cited_paper_id][year]
    ret_val['hub_score'] = PAPER_TO_HUB[cited_paper_id][year]
    ret_val['authority_score'] = PAPER_TO_AUTHORITY[cited_paper_id][year]
    ret_val['load_centrality'] = PAPER_TO_LOAD_CENTRALITY[cited_paper_id][year]

    # What kind of paper is citing?
    citing_paper_id = parsed_doc['paper_id']
    if len(citing_paper_id) == 0:
        print 'HELP', parsed_doc['file_id']
        return {}

    if citing_paper_id[0] == 'J' or citing_paper_id[0] == 'Q':
        ret_val['IsJournal'] = 1
    elif citing_paper_id[0] == 'S' or citing_paper_id[0] == 'W':
        ret_val['IsWorkshop'] = 1
    else:
        ret_val['IsConference'] = 1

    #ret_val['Citing_Venue_' + citing_paper_id[0:1]] = 1

    # What kind of paper is the cited
    if 'External' in cited_paper_id:
        # Try to guess
        ref_str = citation_context['raw_string']
        ret_val[get_citation_type(ref_str)] = 1
        #ret_val['Cited_Venue_External'] = 1        
    else:
        if cited_paper_id[0] == 'J' or cited_paper_id[0] == 'Q':
            ret_val['JournalRef'] = 1
        elif cited_paper_id[0] == 'S' or cited_paper_id[0] == 'W':
            ret_val['WorkshopRef'] = 1
        else:
            ret_val['ConferenceRef'] = 1
        #ret_val['Cited_Venue_' + cited_paper_id[0:1]] = 1        


    DOES_SENTENCE_START_WITH_CITATION = int(sent.startswith(citing_string))

    try:
        citePositionInSentence = sent.index(citing_string) / float(len(sent))
    except:
        print '(%d:%d:%d) %s at %d in %s' % (citation_context['section'], \
                                                 citation_context['subsection'], \
                                                 citation_context['sentence'], \
                                                 citing_string, sent.find(citing_string), sent)

    citePositionInSentence = sent.index(citing_string) / float(len(sent))
        
    # Grab the text right before the citation
    ref_index = sent.find(citing_string)
    sentence_text_before_citation = sent[:ref_index].strip()
    sentence_text_after_citation = sent[(ref_index + len(citing_string)):].strip()   

    # Do so munging to strip off (a) prior citations if this is in the middle of
    # a citation block, (b) trailing citations, (c) parentheses
    i = sentence_text_before_citation.rfind('(')
    if i >= 0:
        sentence_text_before_citation = sentence_text_before_citation[:i].strip()
        
    i = sentence_text_after_citation.find(')')
    if i >= 0:
        sentence_text_after_citation = sentence_text_after_citation[(i+1):].strip()
       
    if "(" in citing_string:
        citing_string_bracket = citing_string.split("(")[1].split(")")[0]
    else:
        citing_string_bracket = citing_string

    NUM_CITATIONS_IN_SAME_CITATION = 0

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

    if preceding_text.endswith('e.g.,') \
            or preceding_text.endswith('e.g.') \
            or preceding_text.endswith('for example') \
            or preceding_text.endswith('for example,') \
            or preceding_text.endswith('for instance') \
            or preceding_text.endswith('for instance,') \
            or preceding_text.endswith('see') \
            or preceding_text.endswith('cf.') \
            or preceding_text.endswith('cfXXX') \
            or preceding_text.endswith('cf'):
        #print 'Saw %s in example in sent: %s' % (citing_string, sent);
        IS_EXAMPLE = 1
    elif is_cite_in_parens and (full_cite.startswith("(e.g.") \
                                    or full_cite.startswith("(cf.") \
                                    or full_cite.startswith("(cfXXX")):
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
    num_citations_in_subsec = 0
    for ssent in subsection['sentences']:
        num_citations_in_subsec += len(re.findall("([0-9][0-9][0-9][0-9])", ssent['text']))
    
    # 
    #topic_features = get_topic_features(subsection['sentences'], sent_index)
    #citation_topics = citation_context['context_topics']
    #extended_citation_topics = citation_context['extended_context_topics']
    #topic_features = OrderedDict()
    #for i, prob in enumerate(citation_topics):
    #    topic_features['Topic_' + str(i)] = prob
    #for i, prob in enumerate(extended_citation_topics):
    #    topic_features['ExtendedTopic_' + str(i)] = prob


    # Search for compare and contrast stuff
    compare_or_contrast_keyword_indicators = {}
    if True:
        compare_or_contrast_keyword_indicators['distance_to_compare'] = 100
        compare_or_contrast_keyword_indicators['distance_to_contrast'] = 100
        contrast_seen = False
        compare_seen = False
        for fw_idx in range(sent_index+1, min(sent_index+5, len(subsection['sentences']))):
            tmpsent = subsection['sentences'][fw_idx]['tokens']
            if not contrast_seen and contains_contrast(tmpsent):
                compare_or_contrast_keyword_indicators['distance_to_contrast'] = fw_idx - sent_index
                contrast_seen = True
            if not compare_seen and contains_compare(tmpsent):
                compare_or_contrast_keyword_indicators['distance_to_compare'] = fw_idx - sent_index
                compare_seen = True
            # Stop early if both found
            if compare_seen and contrast_seen:
                break
    else:
        contrast_seen = False
        compare_seen = False
        for fw_idx in range(sent_index+1, min(sent_index+5, len(subsection['sentences']))):
            tmpsent = subsection['sentences'][fw_idx]['tokens']
            if not contrast_seen and contains_contrast(tmpsent):
                compare_or_contrast_keyword_indicators['contrast_marker_at_' + str(fw_idx - sent_index)] = 1
                contrast_seen = True
            if not compare_seen and contains_compare(tmpsent):
                compare_or_contrast_keyword_indicators['compare_marker_at_' + str(fw_idx - sent_index)] = 1
                compare_seen = True
            # Stop early if both found
            if compare_seen and contrast_seen:
                break



    # Parse the current sentence and the citation
    

    #processed_sent = process_sent(sent.replace(citing_string, "CITATION"))
    replaced_cite_sent = preceding_text + ' ' + CITATION + ' ' + anteceding_text
    #print 'CONVERTING:\n\t%s\n\t%s\n' % (sent, replaced_cite_sent)
    processed_sent = process_sent(replaced_cite_sent)

    dep_path_features = get_dependency_features(replaced_cite_sent, data_to_exclude=parsed_doc['file_id'])

    #processed_cite = process_cite(citing_string)
    
    # Figure out where the citation occurs in the parsing
    cite_index = -1
    relative_clause_cite_position = citePositionInSentence
    sent_length = len(processed_sent)
    for ti, token in enumerate(processed_sent):
        if token['word'].endswith('CITATION'):
            cite_index = ti
            break

    fs_formulaic_features = get_formulaic_features(processed_sent, cite_index, prefix='FullSent:')
    fs_agent_features = get_agent_features(processed_sent, cite_index, prefix='FullSent:')

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

            for ti, token in enumerate(processed_sent):
                if token['word'].endswith('CITATION'):
                    cite_index = ti
                    break

            #print '%s (%d) should be citation in "%s"' \
            #    % (processed_sent[cite_index]['word'], cite_index, to_str(processed_sent))
    else:
        #print "couldn't find '%s' in '%s'" \
        #    % (to_str(processed_cite), to_str(processed_sent))
        pass
    clause_length = len(processed_sent)        

    formulaic_features = get_formulaic_features(processed_sent, cite_index, prefix='InClause:')
    agent_features = get_agent_features(processed_sent, cite_index, prefix='InClause:')
    action_features = get_action_features(processed_sent, cite_index, '')
    concept_features = get_concept_features(processed_sent, cite_index, '')


    processed_prec = processed_sent[cite_index-4:cite_index]
    processed_ante = processed_sent[cite_index+1:cite_index+5]
    
    prec_formulaic = {}
    prec_action = {}
    if len(processed_prec) > 0:
        prec_formulaic = get_formulaic_features(processed_prec, len(processed_prec), prefix='Preceding:')
        prec_action = get_action_features(processed_prec, len(processed_prec), 'Preceding:')
        prec_agent = get_agent_features(processed_prec, len(processed_prec), prefix='Preceding:')
        pass

    ante_formulaic = {}    
    ante_action = {}    
    if len(processed_ante) > 0:
        ante_formulaic = get_formulaic_features(processed_ante, -1, prefix='Following:')
        ante_action = get_action_features(processed_ante, -1, 'Following:')
        ante_agent = get_agent_features(processed_ante, -1, prefix='Following:')


    verb_syntax_feats = {}
    has_root_verb = False
    for token in processed_sent:
        # Find the verb
        if 'is_root' in token:
            if 'is_pass' in token:
                verb_syntax_feats['verb_is_passive'] = 1
            else:
                verb_syntax_feats['verb_is_active'] = 1

            if 'tense' in token:
                verb_syntax_feats['verb_tense_' + token['tense']] = 1

            if 'has_aux' in token:
                verb_syntax_feats['verb_has_modal'] = 1
            else:
                verb_syntax_feats['verb_has_no_modal'] = 1

            has_root_verb = True

    if not has_root_verb:
        verb_syntax_feats['no_verb_for_modal'] = 1
        verb_syntax_feats['no_verb_for_voice'] = 1
        verb_syntax_feats['no_verb_for_tense'] = 1
    




    # Use the 2 sentences before as pre-citance context
    preceding_sents = []
    for prev in range(sent_index, max(-1, sent_index-3), -1):
        preceding_sents.append(subsection['sentences'][prev]['tokens'])

    # Use the 4 sentences before as pre-citance context
    following_sents = []
    for foll in range(sent_index+1, min(sent_index+5, len(subsection['sentences']))):
        following_sents.append(subsection['sentences'][foll]['tokens'])

    custom_pattern_features = get_custom_pattern_features(\
        processed_sent, preceding_sents, following_sents, cite_index)

    connector_features = get_connector_words(sent_index, subsection)

    # Figure out where this citation occurs in the section and subsetion,
    # relatively
    num_sents_in_sec = 0
    num_before_sents_in_sec = 0
    num_sents_in_paper = 0
    num_sents_in_paper_before_cite = 0

    for ss_i, subsec in enumerate(parsed_doc['sections'][citation_context['section']]['subsections']):
        num_sents_in_sec += len(subsec['sentences'])
        if ss_i < citation_context['subsection']:
            num_before_sents_in_sec += len(subsec['sentences'])

    for s_i, sec in enumerate(parsed_doc['sections']):
        for ss_i, ssec in enumerate(sec['subsections']):
            for sss_i, sssent in enumerate(ssec['sentences']):
                num_sents_in_paper += 1
                
                if ss_i <= citation_context['section'] \
                        and ss_i <= citation_context['subsection'] \
                        and sss_i <= citation_context['sentence']:
                    num_sents_in_paper_before_cite += 1

                
    
    relative_pos_in_subsection = float(citation_context['sentence']) \
        / len(subsection['sentences'])
    relative_pos_in_section = float(citation_context['sentence'] + num_before_sents_in_sec) \
        / num_sents_in_sec
    relative_pos_in_paper = float(num_sents_in_paper_before_cite) \
        / num_sents_in_paper

    
    # See if the subsection has a recognizable title (e.g., is the Results
    # subsection of section)
    section_title_feats = {}
    if 'title' in subsection:
        section_title_feats = get_section_title_feature(subsection['title'])
    # If not, use the section's title
    if len(section_title_feats) == 0 and 'title' in section:
        section_title_feats = get_section_title_feature(section['title'])


    ret_val.update(section_title_feats)
    ret_val['Section_Num'] = citation_context['section']
    ret_val['Sections_Left'] = len(parsed_doc['sections']) - citation_context['section']
    ret_val['sentence_length'] = sent_length
    ret_val['relative_pos_in_subsection'] = relative_pos_in_subsection
    ret_val['relative_pos_in_section'] = relative_pos_in_section
    ret_val['relative_pos_in_paper'] = relative_pos_in_paper
    ret_val['clause_length'] = clause_length
    ret_val['IS_USED_AS_TEXT'] = IS_USED_AS_TEXT
    ret_val['is_cite_in_parens'] = IS_CITE_IN_PARENS
    ret_val['DOES_SENTENCE_START_WITH_CITATION'] = DOES_SENTENCE_START_WITH_CITATION
    ret_val['NUM_CITATIONS_IN_CITATION_SENTENCE'] = num_citations_in_sent
    ret_val['NUM_CITATIONS_IN_SUBSECTION'] = num_citations_in_subsec
    ret_val['NUM_CITATIONS_IN_SAME_CITATION'] = NUM_CITATIONS_IN_SAME_CITATION

    ret_val['CitePositionInSentence'] = citePositionInSentence
    ret_val['relative_clause_cite_position'] = relative_clause_cite_position        
    # ret_val['RelativeSentPosition'] = SENTENCE_INDEX / float(len(text_sents))

    ret_val['ALL_CAPS'] = ALL_CAPS
    ret_val['CAMELCASE'] = CAMELCASE
    ret_val['IS_EXAMPLE'] = IS_EXAMPLE
    
    if is_self_cite:
        ret_val['IS_SELF_CITE'] = 1

    ret_val.update(formulaic_features)
    ret_val.update(agent_features)
    ret_val.update(fs_formulaic_features)
    ret_val.update(fs_agent_features)
    ret_val.update(action_features)
    ret_val.update(concept_features)
    ret_val.update(connector_features)
    ret_val.update(custom_pattern_features)                        
    ret_val.update(verb_syntax_feats)
    ret_val.update(compare_or_contrast_keyword_indicators)

    ret_val.update(prec_formulaic)
    ret_val.update(ante_formulaic)
    ret_val.update(prec_action)
    ret_val.update(ante_action)

    # ret_val.update(topic_features)
    ret_val.update(dep_path_features)

    if False and citation_context['citation_role'].startswith('Future'):
        print '%s, %s:\n\tsentece: %s\n\tclause: %s' % (citation_context['citation_role'], citing_string, sent, to_str(processed_sent))
        for k, v in ret_val.iteritems():
            f = float(v)
            z = int(v)
            s = str(v)
            if s != '0' and  z != 100:
                print '\t%s\t%s' % (k, s)
        print ''

    # print sent
    # print ret_val
    return ret_val

def get_topic_features(sentences, sent_index):
    topic_feats = {}
    lemmas = []
    for t in sentences[sent_index]['tokens']:
        lemmas.append(t['lemma'])
    vec = LDA_CITANCE_ONLY_DICT.doc2bow(lemmas)
    citance_topics = LDA_CITANCE_ONLY_MODEL[vec]
    for t in citance_topics:
        topic_feats['Citance_Topic_' + str(t[0])] = t[1]

    lemmas = []
    for s in sentences[sent_index-1:sent_index+4]:
        for t in s['tokens']:
            lemmas.append(t['lemma'])
    vec = LDA_EXTENDED_CONTEXT_DICT.doc2bow(lemmas)
    citance_topics = LDA_EXTENDED_CONTEXT_MODEL[vec]
    for t in citance_topics:
        topic_feats['ExCon_Topic_' + str(t[0])] = t[1]

    return topic_feats

def get_connector_words(sent_index, subsec):
    seen_connectors = {}

    for c in CONNECTORS:
        seen_connectors[c] = 100

    for fw_idx in range(sent_index, min(sent_index, len(subsec['sentences']))):

        sent = subsec['sentences'][fw_idx]['text'][:15].lower()
        for c in CONNECTORS:
            if sent.startswith(c):
                seen_connectors['Connector_'  +c] = (fw_idx - sent_index) + 1

    return seen_connectors

def get_citation_type(ref_str):
    if 'Workshop' in ref_str:
        return 'WorkshopRef'
    elif 'Proceedings' in ref_str:
        return 'ConferenceRef'
    elif 'Proc.' in ref_str:
        return 'ConferenceRef'
    elif 'Journal' in ref_str:
        return 'JournalRef'
    # Look for the (20)1 notation for volume/number
    elif len(re.findall("\(\d+\)\d", ref_str)) > 0:
        return 'JournalRef'
    else:
        return 'UnknownRefType'


def count_indirect_citations(citing_strings, cite_specific_phrases, \
                                 sents_to_exclude, annotated_data):
    total_ind_cites = 0
    ind_cites_per_sec = Counter()

    irefs = set()
    for cstr in citing_strings:
        iref = re.sub(r'[0-9.,();]', '', cstr).strip()
        irefs.add(iref)

    for sec_i, section in enumerate(annotated_data['sections']):
        for sub_i, subsec in enumerate(section['subsections']):

            # Get the working title for where we are in the paper
            cur_section_title = None
            if 'title' in subsec:
                section_title_feats = get_section_title_feature(subsec['title'])
                if len(section_title_feats) > 0:
                    cur_section_title = section_title_feats.keys()[0]
            # If no usable subsection title, use the section's title
            if cur_section_title is None and 'title' in section:
                section_title_feats = get_section_title_feature(section['title'])
                if len(section_title_feats) > 0:
                    cur_section_title = section_title_feats.keys()[0]
            # If neither has an informative title, put this in "other"
            if cur_section_title is None:
                cur_section_title = 'OtherSection'

            for sent_i, sent in enumerate(subsec['sentences']):

                # Skip sentences that we know the citation occurs in
                if (sec_i, sub_i, sent_i) in sents_to_exclude:
                    continue
                else:
                    # See if any of the possible indirect references occurs
                    found = False
                    for iref in irefs:
                        i = sent['text'].find(iref)
                        if i >= 0:
                            # Check this isn't a part of another ref by looking
                            # for a year immediately after where the indirect
                            # reference is found
                            if re.search(r'[0-9]{4}', sent['text'][i+len(iref):i+len(iref)+25]) is None:
                                #print 'Saw indirect citation "%s" in sent: %s' % (iref, sent['text'])
                                total_ind_cites += 1
                                ind_cites_per_sec[cur_section_title] += 1
                                found = True
                                break
                    if found:
                        continue

                    # NB: Probably should be explicitly checking for word
                    # boundaries here, but assumption is phrase is specific
                    # enough that it's not a substring of other stuff
                    for phrase in cite_specific_phrases:
                        if phrase in sent['text']:
                            #print 'Saw phrasal indirect citation "%s" in sent: %s' % (phrase, sent['text'])
                            #print phrase
                            total_ind_cites += 1
                            ind_cites_per_sec[cur_section_title] += 1
                            break
                        

    return total_ind_cites, ind_cites_per_sec

def sim(word1, word2):
    if (word1 not in WORD_TO_VEC) or (word2 not in WORD_TO_VEC):
        return 0
    vec1 = WORD_TO_VEC[word1]
    vec2 = WORD_TO_VEC[word2]
    return 1 - spatial.distance.cosine(vec1, vec2)

RAW_PATH_DATA = []
ALL_GOLD_PATH_DATA = defaultdict(lambda: defaultdict(list))
ALL_GOLD_PATH_VECS = defaultdict(dict)
def load_path_data():

    # path_to_gold_leaves = defaultdict(lambda: defaultdict(list))

    with open('../resources/patterns/dependency-contexts.tsv') as f:
        for line in f:
            cols = line.split('\t')
            fname = cols[0]
            label = cols[1]
            path_elements = cols[2].split()
            if len(path_elements) == 3:
                path = (path_elements[0],)
                leaf = path_elements[1]
            else:
                path = (path_elements[0], path_elements[1])
                leaf = path_elements[2]
            if 'punct' in path or 'det' in path:
                continue

            #print 'PATH2: ', path
            
            RAW_PATH_DATA.append((fname, label, path, leaf))
            ALL_GOLD_PATH_DATA[path][label].append(leaf)

    # Aggregate the vectors
    for path, labels in ALL_GOLD_PATH_DATA.iteritems():
        for label, fillers in labels.iteritems():
            # TODO: make this dependent on vec size
            avg_vec = np.zeros(300)
            
            added = False
            for filler in fillers:
                if filler in WORD_TO_VEC:
                    avg_vec += WORD_TO_VEC[filler]
                    added = True
            if added:
                ALL_GOLD_PATH_VECS[path][label] = avg_vec
load_path_data()            

# So we don't have to construct the path data for each citation, only once per
# paper with the cache
PATH_DATA_CACHE = {}

def construct_path_data(path_data_to_exclude):
    
    if path_data_to_exclude in PATH_DATA_CACHE:
        return PATH_DATA_CACHE[path_data_to_exclude]

    path_to_gold_leaves = defaultdict(lambda: defaultdict(list))

    excluded = 0
    for fname, label, path, leaves in RAW_PATH_DATA:
        # print '"%s" ?= "%s" : %s' % (fname, path_data_to_exclude, fname == path_data_to_exclude)
        if fname == path_data_to_exclude:
            excluded += 1
            continue
        path_to_gold_leaves[path][label].append(leaves)


    path_to_gold_vecs = defaultdict(dict)

    # Aggregate the vectors
    for path, labels in path_to_gold_leaves.iteritems():
        for label, fillers in labels.iteritems():
            # TODO: make this dependent on vec size

            added = False
            avg_vec = np.zeros(300)

            for filler in fillers:
                if filler in WORD_TO_VEC:
                    avg_vec += WORD_TO_VEC[filler]
                    added = True
            
            if added:
                path_to_gold_vecs[path][label] = avg_vec

    PATH_DATA_CACHE[path_data_to_exclude] = (path_to_gold_leaves, path_to_gold_vecs)

    # print 'Excluded %d items from %s' % (excluded, path_data_to_exclude)
    return path_to_gold_leaves, path_to_gold_vecs

def get_dependency_features(replaced_cite_sent, data_to_exclude=None):
    
    doubly_replaced_cite_sent = re.sub(r'\([^)]*[0-9]{4}[^)]*\)', 'CITANCE', replaced_cite_sent)
    # strip off multiple citation cruft
    doubly_replaced_cite_sent = re.sub(r'([, ]*[0-9]{4}[, ]*)+\)', '', doubly_replaced_cite_sent)
    processed_sent, deps = process_sent(doubly_replaced_cite_sent, get_deps=True)
        
    # Figure out where the citation is in the sentence now
    path_to_leaves = {}
    cite_found = False
    for i, token in enumerate(processed_sent):
        if token['word'].endswith('CITATION'):
            # Generate the dependency paths from the citation
            path_to_leaves = get_paths(processed_sent, deps, i)
            cite_found = True
            break
    if not cite_found:
        # print 'Unable to find CITATION in ', to_str(processed_sent)
        pass

    if data_to_exclude is not None:
        path_to_gold_leaves, path_to_gold_vecs = construct_path_data(data_to_exclude)
    else:
        path_to_gold_leaves = ALL_GOLD_PATH_DATA
        path_to_gold_vecs = ALL_GOLD_PATH_VECS

    #print 'saw %d paths to gold leaves and %d vecs' % (len(path_to_gold_leaves), len(path_to_gold_vecs))
    
    labled_path_to_scores = defaultdict(lambda: defaultdict(list))

    # print 'saw %d paths to leaves' % len(path_to_leaves)

    if False:
        # print 'Saw %d paths with %d leaves' % (len(path_to_leaves), sum(len(x) for x in path_to_leaves.values()))
        for path, leaves in path_to_leaves.iteritems():
            #print 'PATH: ',  path
            if not path in path_to_gold_leaves:
                # print '%s not in %s' % (str(path), str(path_to_gold_leaves.keys()))
                continue
            for label, gold_leaves in path_to_gold_leaves[path].iteritems():
                for l1 in leaves:
                    for l2 in gold_leaves:
                        labled_path_to_scores[label][path].append(sim(l1, l2))
    else:
        num_missing_fillers = 0
        num_fillers = 0
        for path, leaves in path_to_leaves.iteritems():
            print path
            if not path in path_to_gold_leaves:
                #print '%s not in %s' % (str(path), str(path_to_gold_leaves.keys()))
                # print '\tMISSING!'
                continue
            
            # print 'Saw %d gold vecs at end of path %s' % (len(path_to_gold_vecs[path]), path)

            for label, gold_vec in path_to_gold_vecs[path].iteritems():
                # print '\tPRESENT!'
                num_fillers += 1
                for filler in leaves:
                    if filler not in WORD_TO_VEC:
                        num_missing_fillers += 1
                        continue
                    filler_vec = WORD_TO_VEC[filler]
                    vec_sim = 1 - spatial.distance.cosine(gold_vec, filler_vec)
                    if math.isnan(vec_sim):
                        vec_sim = 0
                    labled_path_to_scores[label][path].append(vec_sim)
            
        #print 'missing vectors for %d/%d fillers' % (num_missing_fillers, num_fillers)
    pattern_feature_vals = {}

    for label, path_to_scores in labled_path_to_scores.iteritems():
        scores_sum = 0
        num_scores = 0
        # print 'saw %d path scores for %s' % (len(path_to_scores), label)
        for path, scores in path_to_scores.iteritems():
            feature = 'PATH:' + label + '_' + '_'.join(path)
            scores.sort()
            feature_val = sum(scores[-3:]) / float(min(3, len(scores)))
            #pattern_feature_vals[feature] = feature_val
            #print feature, ' -> ', feature_val
            scores_sum += sum(scores)
            num_scores += len(scores)

        if num_scores > 0:
            pattern_feature_vals[label + "_sel_pref_sim"] = scores_sum / num_scores

    print ''
    #print pattern_feature_vals

    return pattern_feature_vals

def get_paths(tokens, deps, origin):
    tree = defaultdict(list)
    intree = defaultdict(list)

    # Build the tree
    for dep in deps:
        if dep['dep'] == 'ROOT':
            continue
        # tree['ROOT'].append(('root', dep['dependent']))
        #    intree[dep['dependent']].append(('inv-root', 'ROOT'))
        else:
            gov_id = dep['governor']
            dep_id = dep['dependent']
            if dep['dep'] == 'punct':
                if dep_id == len(tokens):
                    continue

            dep_id -= 1
            gov_id -= 1

            tree[gov_id].append((dep['dep'], dep_id))
            intree[dep_id].append(('inv-' + dep['dep'], gov_id))
    


    paths = []

    # Get the outgoing edges from the origin
    for dep, dep_id in tree[origin]:
        n = tokens[dep_id]
        p1 = (True, dep, n['lemma'])
        paths.append([p1])
        
        # Expand any outgoing paths from p1
        for dep2, dep_id2 in tree[dep_id]:
            n2 = tokens[dep_id2]
            p2 = (True, dep2, n2['lemma'])
            paths.append([p1, p2])            

        # Expand any incoming paths to p1 excluding the origin
        for dep2, dep_id2 in intree[dep_id]:
            if dep_id2 == origin:
                continue
            n2 = tokens[dep_id2]
            p2 = (False, dep2, n2['lemma'])
            paths.append([p1, p2])            

    # Get the incoming edges to the origin
    for dep, dep_id in intree[origin]:
        n = tokens[dep_id]
        p1 = (False, dep, n['lemma'])
        paths.append([p1])
        
        # Expand any outgoing paths from p1 excluding the origin
        for dep2, dep_id2 in tree[dep_id]:
            if dep_id2 == origin:
                continue

            n2 = tokens[dep_id2]
            p2 = (False, dep2, n2['lemma'])
            paths.append([p1, p2])            

        # Expand any incoming paths to p1
        for dep2, dep_id2 in intree[dep_id]:
            n2 = tokens[dep_id2]
            p2 = (True, dep2, n2['lemma'])
            paths.append([p1, p2])            


    # TODO: convert the paths into some kind of canonical wizardry
            
    # type1: dep/dir -> word/pos
    # type2: dep/dir -> dep/dir -> word/pos

    path_to_leaves = defaultdict(list)
    for p in paths:
        if len(p) == 1:
            path_to_leaves[tuple(p[0][1:-1])].append(p[0][-1])
        else:
            tmp = [p[0][1]]
            tmp.extend(p[1][1:-1])
            tmp = tuple(tmp)
            path_to_leaves[tmp].append(p[1][-1])

    #for p, c in path_counts.iteritems():
    #    print p, c

    return path_to_leaves
