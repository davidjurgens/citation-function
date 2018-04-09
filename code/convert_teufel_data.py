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
from convert_ARC_xml_to_json import process_sent
from convert_ARC_xml_to_json import get_paper_id
#from global_functions_march16 import *
import os
from collections import defaultdict
from collections import Counter
from joblib import Parallel, delayed
import fnmatch
from lxml import etree
import json
from pycorenlp import StanfordCoreNLP

corenlp = StanfordCoreNLP('http://localhost:9000')

INPUT_DIR='../data/teufel/CFC_distribution/'

def main():

    if len(sys.argv) < 2:
        print 'Usage: convert_ARC_to_features.py out-dir/ [input-dir]'
        return

    output_dir = sys.argv[1]
    input_dir = INPUT_DIR
    if len(sys.argv) > 2:
        input_dir = sys.argv[2]

    xml_files = []
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, '*.cfc-scixml'):
            if '9502039' not in filename:
                #continue
                pass
            xml_files.append(os.path.join(root, filename))
            
    #[process(fname, output_dir) for fname in xml_files]    
    Parallel(n_jobs=64)(delayed(process)(fname, output_dir) for fname in xml_files)

def process(fname, output_dir):

    
    xmldoc = minidom.parse(fname)

    ref_list = xmldoc.getElementsByTagName("REFERENCELIST")
    
    ref_list = xmldoc.getElementsByTagName("REFERENCE")


    paper_info = get_paper_info(xmldoc)

    if paper_info['year'] < 0:
        #print 'No year in ' + fname
        
        # this only happens in one case, for which we guess using the filename
        # (which was verified to be correct)
        year = int(basename(fname)[0:2]) 
        if year < 50:
            year += 2000
        else:
            year += 1900
        paper_info['year'] = year
        

    if paper_info['year'] < 1000:
        year = paper_info['year']
        if year < 50:
            year += 2000
        else:
            year += 1900
        paper_info['year'] = year
        
    refs = []

    # Ughhhh, year is typed as str
    paper_info['title'] = paper_info['title'].replace(":", "").replace('?', '')\
        .replace('"', "").replace(',', "").replace('.', "")\
        .replace('\'', "").replace('`', "")\
        .replace('  ', ' ').strip()

    # Some of the titles are impossible to match with the ARC, so we just hard
    # code them in Teufel's XML
    paper_id = get_hardcoded_paper_id(xmldoc)

    if paper_id == '':
        paper_id = get_paper_id(paper_info['title'].lower(), str(paper_info['year']), \
                                  paper_info['authors'])
        if paper_id == '':
            title2 = paper_info['title'].replace("-", "").replace('  ', ' ')
            paper_id = get_paper_id(title2.lower(), str(paper_info['year']), \
                                      paper_info['authors'])
            paper_info['title'] = title2
    
    if paper_id == '' or paper_id.startswith("Ext"):
        #print 'No ID for %s :: %s (%d) "%s"' % (basename(fname), str(paper_info['authors']),\
        #                                          paper_info['year'], paper_info['title'].lower())
        pass
    else:
        #print "%s -> %s" % (basename(fname), paper_id)
        pass

    for ref in ref_list:
        names = []
        date = None
        for i, c in enumerate(ref.childNodes):
            text = c.toprettyxml().strip()
            if 'DATE' in text:
                date = text
            elif date is None and len(text) > 0:
                names.append(text.replace('\n', ' '))
            # print 'child %d (%s): %s' % (i, c.nodeType, )

        raw_str = ref.toprettyxml().strip().replace('\n', ' ').replace('  ', ' ')
        raw_str = re.sub(r'\s+', ' ', raw_str)
        i = raw_str.find('</DATE>')
        raw_str = raw_str[i+9:]
        j = raw_str.find('.')
        title = raw_str[:j].strip()

        ref_str = ref.toprettyxml().strip().replace('\n', ' ').replace('  ', ' ')
        ref_xml = etree.fromstring(ref_str)
        etree.strip_tags(ref_xml, 'REFERENCE')
        etree.strip_tags(ref_xml, 'SURNAME')
        etree.strip_tags(ref_xml, 'REFLABEL')
        etree.strip_tags(ref_xml, 'DATE')
        ref_str = re.sub(r'\s+', ' ', etree.tostring(ref_xml))[11:-13].strip()
        
        #print ref_str
        
        cleaned_names, surnames = clean_names(names)
        #print cleaned_names
        #print '' 
        #print '' 
        
        # This happens with empty reference tags
        if date is None:
            # print ref.toprettyxml().strip()
            continue
        
        # We're going to have to guess, so pick the publication date of the
        # citing paper
        if 'ppear' in date or 'press' in date or 'orthcoming' in date or 'in print' in date:
            date = paper_info['year']
        else:
            try:
                date = int(date[6:10])
            except ValueError:
                print date

        refs.append({ 'authors': cleaned_names, 'surnames': surnames, 'year': int(date), 'raw_string': ref_str, 'title': title })
        #print 'names: %s\ncleaned_names: %s\ndate: %s\nref_str: %s' % (str(names), str(cleaned_names), date, ref_str)


    # Assign them to ARC IDs (hopefully...)
    canonicalize_refs(refs)

    prev_sec_to_name = None #cur_section
    sections = []
    citation_contexts = []
    body = xmldoc.getElementsByTagName("BODY")[0]

    for node in body.childNodes:
        if node.nodeName == 'DIV':
            section = process_section(node, refs, citation_contexts, len(sections))
            section['num'] = len(sections)
            # print 'Section %d had %d subsections' % (section['num'], len(section['subsections']))
            sections.append(section)

    parsed_doc = { 'sections': sections, 'citation_contexts': citation_contexts }
    parsed_doc['year'] = paper_info['year']
    parsed_doc['paper_id'] = paper_id
    parsed_doc['authors'] = paper_info['authors']
    parsed_doc['title'] = paper_info['title']

    # Canonicalize the citation IDs
    for i, cc in enumerate(citation_contexts):
        cc['citation_id'] = 'Teufel_' + basename(fname) + "_" + str(i)

    
    outfilename = output_dir + '/'  + basename(fname) + ".json"
    with open(outfilename, 'w') as outf:
        outf.write(json.dumps(parsed_doc))
        outf.write('\n')

    print 'finished ', fname

def get_hardcoded_paper_id(xmldoc):
    nodes = xmldoc.getElementsByTagName("PAPER_ID")
    if nodes is not None and len(nodes) > 0:
        return nodes[0].firstChild.nodeValue
    else:
        return ''

def canonicalize_refs(refs):
    for ref in refs:
        title = ref['title'].replace(":", "").replace('?', '')\
            .replace('"', "").replace(',', "").replace('.', "")\
            .replace('\'', "").replace('`', "")\
            .replace('  ', ' ').strip()
        
        paper_id = get_paper_id(title.lower(), str(ref['year']), \
                                  ref['authors'])
        if paper_id == '':
            title2 = title.replace("-", "").replace('  ', ' ')
            paper_id = get_paper_id(title2.lower(), str(ref['year']), \
                                      ref['authors'])
        
        # Assign it a new ID
        if paper_id == '':
            ref['paper_id'] = 'External_DUMMY'
        else:
            #print 'Actually found a ref: %s <- %s' % (paper_id, title)
            ref['paper_id'] = paper_id

def get_paper_info(xmldoc):
    paper_info = defaultdict(str)
    metadata = xmldoc.getElementsByTagName("METADATA")[0]
    
    for node in metadata.childNodes:
        if node.nodeName == 'APPEARED':
            for n2 in node.childNodes:
                # print n2.nodeName
                if n2.nodeName == 'YEAR':
                    if n2.firstChild is not None:
                        paper_info['year'] = int(n2.firstChild.nodeValue)
            if node.firstChild is None or node.firstChild.nodeValue is None:
                continue
            elif node.firstChild.nodeValue[-4:].isdigit():
                paper_info['year'] = int(node.firstChild.nodeValue[-4:])
            else:
                try:
                    year = int(node.firstChild.nodeValue.strip()[-2:])
                except ValueError:
                    numbers = [int(s) for s in re.findall(r'\d+', node.firstChild.nodeValue) ]
                    if len(numbers) > 0:
                        year = numbers[0]
                    else:
                        year = -9000
                    # print 'Choosing %d for year in "%s"' % (year, node.firstChild.nodeValue)
                
                if year < 50:
                    year += 2000
                elif year < 1900:
                    year += 1900
                paper_info['year'] = year



    if paper_info['year'] == '':
        print 'unknown year!'
        paper_info['year'] = -1
    paper = xmldoc.getElementsByTagName("PAPER")[0]
    
    for node in paper.childNodes:
        if node.nodeName == 'TITLE':
            paper_info['title'] = node.firstChild.nodeValue.strip()
        elif node.nodeName == 'AUTHORLIST':
            authors = []
            for n2 in node.childNodes:
                if n2.firstChild is not None:
                    authors.append(n2.firstChild.nodeValue.strip())
            paper_info['authors'] = authors

    return paper_info


def process_section(sec_node, refs, citation_contexts, cur_sec_num):
    cur_subsection = { 'sentences': [ ], 'title': '', 'num': 0}
    cur_section = { 'subsections': [ cur_subsection ], 'num': cur_sec_num, 'title': ''}

    for node in sec_node.childNodes:
        
        # New Subsection
        if node.nodeName == 'DIV':
            cur_subsection = process_subsection(node, refs, citation_contexts, cur_sec_num, len(cur_section['subsections']))
            cur_subsection['num'] = len(cur_section['subsections'])
            cur_section['subsections'].append(cur_subsection)
            #print 'Sec %d now has %d subsecs' % (cur_sec_num, len(cur_section['subsections']))
        
        elif node.nodeName == 'HEADER':
            #print '%s -> %s' % (node.getAttribute('ID'), node.firstChild.nodeValue)
            cur_section['title'] = node.firstChild.nodeValue.strip()


        elif node.nodeName == 'P':
            get_sentences(node, refs, citation_contexts, cur_sec_num, cur_subsection['num'], cur_subsection['sentences'])

    return cur_section


def process_subsection(subsec_node, refs, citation_contexts, cur_sec_num, cur_subsec_num):
    cur_subsection = { 'sentences': [ ], 'title': '' }

    for node in subsec_node.childNodes:
        
        # New Subsection
        if node.nodeName == 'DIV':
            walk_and_append(node, refs, citation_contexts, cur_subsection['sentences'], cur_sec_num, cur_subsec_num)
            

        if node.nodeName == 'HEADER':
            #print '%s -> %s' % (node.getAttribute('ID'), node.firstChild.nodeValue)
            cur_subsection['title'] = node.firstChild.nodeValue.strip()

        elif node.nodeName == 'P':
            get_sentences(node, refs, citation_contexts, cur_sec_num, cur_subsec_num, cur_subsection['sentences'])

    return cur_subsection

def walk_and_append(subsubsec_node, refs, citation_contexts, sentences, cur_sec_num, cur_subsec_num):

    for node in subsubsec_node.childNodes:
        
        # New Subsection
        if node.nodeName == 'DIV':
            walk_and_append(node, refs, citation_contexts, sentences,  cur_sec_num, cur_subsec_num)

        elif node.nodeName == 'P':
            get_sentences(node, refs, citation_contexts, cur_sec_num, cur_subsec_num, sentences)


def get_sentences(p_node, refs, citation_contexts, cur_sec_num, cur_subsec_num, sentences):

    for node in p_node.childNodes:
        if node.nodeName == 'S':
            sent_str = clean_sent(node)

            #print sent_str
            #xml_str = node.toprettyxml()
            #xml_str = xml_str.replace("<EQN/>", "equation")
            #sent = etree.fromstring(xml_str)           

            #etree.strip_tags(sent,'REF')
            #etree.strip_tags(sent, 'REFAUTHOR')
            #sent_str = re.sub(r'\s+', ' ', etree.tostring(sent))
            #i = sent_str.find('>')
            #sent_str = sent_str[(i+2):-5]

            # TODO: fix the parens for cited authors


            #print sent_str              
            #sentences.append(sent_str)
            try:
                parsed_sents = json.loads(corenlp.annotate(sent_str.encode('utf-8'), \
                     properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }).encode('utf-8'), strict=False)

                if len(parsed_sents['sentences']) > 1:
                    # Ughhh.  These are only support to be single sentences. Try
                    # stripping out the periods and manually add a last one
                    sent_str2 = sent_str.replace('.', ' ').replace('!', ' ').replace('?', ' ') + '.'
                    # print 'Replaced\n\t%s\n\twith: %s' % (sent_str, sent_str2)
                    parsed_sents = json.loads(corenlp.annotate(sent_str2.encode('utf-8'), \
                         properties={'annotators': 'tokenize,ssplit,pos,lemma,depparse' }).encode('utf-8'), strict=False)
            except ValueError:
                continue

            cur_sent_num = len(sentences)
            sent = parsed_sents['sentences'][0]
            processed_sent = process_sent(sent)
            sentences.append(processed_sent)           

            # print '%d : %d : %d' % (cur_sec_num, cur_subsec_num, cur_sent_num)
            citing_string = None 
           
            for node2 in node.childNodes:

                ref_obj = None

                if node2.nodeName == 'REF':
                    ref = node2.firstChild.nodeValue
                    func = node2.getAttribute('CFunc')
                    is_self_cite = node2.getAttribute('SELF') == 'YES'
                    #print node2.toprettyxml().strip().replace('\n', ' ')
                    #print '%s -> %s' % (ref, func)
                    
                    ref_obj = find_ref(refs, ref, True)
                    # Skip citations with no function
                    if func == '' or ref_obj is None:
                        continue
                    func = convert_func(func)
                    citing_string = clean_cite(ref, node2.getAttribute('TYPE'))

                if node2.nodeName == 'REFAUTHOR':
                    #print node2.toprettyxml().strip().replace('\n', ' ')
                    ref = node2.firstChild.nodeValue

                    func = node2.getAttribute('CFunc')
                    # Skip citations with no function
                    if func == '':
                        continue

                    is_self_cite = node2.getAttribute('SELF') == 'YES'
                    #print '%s -> %s' % (ref, func)
                    
                    ref_obj = find_ref(refs, ref, False)

                    # In some unusual cases, the REFAUTHOR tag will be used to
                    # discuss an author in a vague sense with an explicit
                    # citation in another REF tag (cf. 9405022.xml); skip these
                    # cases
                    if ref_obj is None:
                        continue

                    func = convert_func(func)
                    citing_string = '%s ( 0000 )' % (ref)
                
                if ref_obj is not None:

                    estimated_type = role_to_type(func)


                    # Ensure we can find the citatin
                    if not citing_string is None:
                        reformatted_sent = processed_sent['text']
                        if not citing_string in reformatted_sent:
                            citing_string = citing_string.replace('.', '')
                            if not citing_string in reformatted_sent:
                                raise BaseException('Unable to find "%s" in "%s"' % (citing_string, reformatted_sent))


                    # Add this as an instance of the current reference
                    # print '%d : %d : %d' % (cur_sec_num, cur_subsec_num, cur_sent_num)
                    context = {
                        'info': ref_obj,
                        'section': cur_sec_num,
                        'subsection': cur_subsec_num,
                        'sentence': cur_sent_num,
                        'citing_string': citing_string,
                        'is_self_cite': is_self_cite,
                        'raw_string': ref_obj['raw_string'],
                        'cite_context': sent_str,
                        'citation_role': func,
                        'citation_type': estimated_type,
                        'cited_paper_id': ref_obj['paper_id'],
                        }
                    #print json.dumps(context)
                    citation_contexts.append(context)
                


    return sentences

def clean_sent(node):
    sent = []
    nodes = node.childNodes
    for i, n in enumerate(nodes):
        if n.nodeName == '#text':
            sent.append(n.data)
        elif n.nodeName == 'REF':
            t = n.getAttribute("TYPE")
            if t == 'A':
                cite = n.firstChild.nodeValue
                cite = re.sub(r'([0-9]{4}[abcde]?)', r'(\1)', cite)
                sent.append(cite)
            else:
                # now the fun starts
                if i > 2 and nodes[i-2].nodeName == 'REF' and nodes[i-1].data.strip() == ',':
                    sent.append(n.firstChild.nodeValue)
                else:
                    sent.append('( ' + n.firstChild.nodeValue)

                if i < len(nodes)-2 and nodes[i+2].nodeName == 'REF' and nodes[i+1].data.strip() == ',':
                    pass
                else:
                    sent[-1] = sent[-1] + ' )'
                    
        elif n.nodeName == 'REFAUTHOR':
            ref = n.firstChild.nodeValue
            sent.append(ref + ' (0000)')
        elif n.nodeName == 'EQN':
            sent.append('equation')
        elif n.nodeName == 'CREF':
            sent.append('CREF')
    return ' '.join(sent)

def role_to_type(role):
    if role == 'Background' or role == 'Motivation' \
            or role == 'Future' or role == 'CompareOrContrast':
        return 'Positional'
    else:
        return 'Essential'

def find_ref(refs, cite_str, has_year):
    is_et_al = 'et al.' in cite_str
    names_in_cite = cite_str.replace(' et al.','').replace(' and', '').split()
    year = 0
    version = ''
    if has_year:
        try:
            if 'appear' in names_in_cite[-1]:
                has_year = False
            else:
                year = int(names_in_cite[-1])
        except ValueError:
            try:
                # This happens when the author name and year are merged Tesar1996
                if names_in_cite[-1][-4:].isdigit():
                    year = int(names_in_cite[-1][-4])
                # This happens when it's like 1997b
                else:
                    year = int(names_in_cite[-1][:-1])
                    version = names_in_cite[-1][-1]
            # These just look like bad data cases
            except ValueError:
                has_year = False
        names_in_cite = names_in_cite[:-1]
    #print '%s -> %s' % (cite_str, str(names))

    found = False
    times_seen = 0
    match = None

    for ref in refs:
        #last_names = [ s.split()[-1] for s in ref['names'] ]
        last_names = ref['surnames']
        
        #print "%s ?= %s :: %s" % (names_in_cite, last_names, last_names == names_in_cite)

        if last_names == names_in_cite \
                or (is_et_al and len(last_names) > 0 and len(names_in_cite) > 0 \
                        and last_names[0] == names_in_cite[0] and len(last_names) >= 3):
            # print 'saw tentative match with %s (%s) ==> %s' % (cite_str, str(names_in_cite), str(ref))
            if has_year:
                if year == ref['year']:
                    times_seen += 1
                    if version_equals(times_seen, version):
                        match = ref
            else:
                times_seen += 1
                if version_equals(times_seen, version):
                    match = ref

    if match is None and has_year:
        #print 'Could not find ref for %s (%s) in \n%s' % (cite_str, str(names_in_cite), str(refs))
        pass

    # print ''
    return match

def version_equals(num, let):
    return let == '' \
        or (num == 1 and let == 'a') \
        or (num == 2 and let == 'b') \
        or (num == 3 and let == 'c') \
        or (num == 4 and let == 'd') \
        or (num == 5 and let == 'e')



def convert_func(func):
    if func == 'Weak':
        return 'CompareOrContrast'
    elif func == 'CoCoGM':
        return 'CompareOrContrast'
    elif func == 'CoCo-':
        return 'CompareOrContrast'
    elif func == 'CoCoR0':
        return 'CompareOrContrast'
    elif func == 'CoCoXY':
        return 'Background'
    elif func == 'PBas':
        return 'Extends'
    elif func == 'PUse':
        return 'Uses'
    elif func == 'PModi':
        return 'Extends'
    elif func == 'PMot':
        return 'Motivation'
    elif func == 'PSim':
        return 'CompareOrContrast'
    elif func == 'PSup':
        return 'CompareOrContrast'
    elif func == 'Neut':
        return 'Background'
    # What are these?
    elif func == 'CoMetN':
        return 'CompareOrContrast'
    elif func == 'CoGoaN':
        return 'CompareOrContrast'
    elif func == 'CoMet-':
        return 'CompareOrContrast'
    elif func == 'CoCoN' or 'CoCoM':
        return 'CompareOrContrast'    
    elif func == 'CoResN':
        return 'CompareOrContrast'    
    else:
        raise BaseException('Unknown function: "%s"' % (func))

def clean_cite(cite_str, ref_type):
    if ref_type == 'P' or ref_type == '':
        #return '(' + cite_str + ')'
        return re.sub(r'([()])', r' \1 ', cite_str)
    elif ref_type == 'A':
        return re.sub(r'([0-9]{4}[abcde]?)', r'( \1 )', cite_str)
    else:
        raise BaseException('unknown ref_type: "%s"' % (ref_type))
            

def clean_names(noisy_names):
    noisy_names = list(noisy_names)
    surnames = []
    names = []
    cur_name = ''
    for i in range(0, len(noisy_names)):
        name = noisy_names[i]
        #print i, name
        if i == 0:
            if 'SURNAME' in name:
                surname = name[9:-10]
                surnames.append(surname)
                if i+1 == len(noisy_names):
                    break
                #print '%d ?= %d: %s' % (i+1, len(noisy_names), names)
                next_name = noisy_names[i+1][2:]
                #print next_name
                # [2:]
                #i += 1
                #parts = next_name.split(',')
                j = next_name.find(',')
                k = 0
                if j < 0:
                    j = next_name.find('and')
                    k = 2
                cur_name = next_name[0:j] + ' ' + surname
                #print '%s -> %s' % (name, cur_name)
                names.append(cur_name)
                noisy_names[i+1] = next_name[j:]
                cur_name = ''
                
            elif ':' in name:
                j = name.find(' ')
                cur_name = name[j+1:]
            elif name[-1].isdigit():
                continue
            else:
                cur_name = name
                
        elif 'SURNAME' in name:
            surname = name[9:-10]
            #print 'SURNAME: ', surname

            # Sanity check for MWE surnames, which happen and are tagged as
            # separate tokens
            for j in range(i + 1, len(noisy_names)):
                nn = noisy_names[j]
                if 'SURNAME' in nn:
                    surname += ' ' + nn[9:-10]
                    noisy_names[j] = '.'
                else:
                    break

            surnames.append(surname)
            cur_name += ' ' + surname
            #print '%s -> %s' % (name, cur_name)
            names.append(cur_name)
            cur_name = ''
        elif name.startswith(', and'):
            cur_name = name[6:]            
        elif name.startswith('and'):
            cur_name = name[4:]            
        elif name.startswith(', '):
            name = name[2:]
            cur_name += ' ' + name
        elif name.startswith(', editor'):
            pass
        elif name.startswith('editor'):
            pass
        elif name.startswith('.'):
            pass
        else:
            #print 'UNKNOWN CASE: %d: %s' % (i, name)
            pass
    if len(cur_name) > 0:
        # print '%s -> %s' % (name, cur_name)
        names.append(cur_name)

    for i, name in enumerate(names):
        while name.startswith(' '):
            name = name[1:]
        names[i] = name

    return names, surnames
    

if __name__ == "__main__":
    main()

