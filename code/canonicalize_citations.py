from xml.dom import minidom
import nltk
from glob import glob
from os.path import basename
import unicodecsv
import string
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import sys
from xml.parsers.expat import ExpatError
from collections import defaultdict
import re
from HTMLParser import HTMLParser
from xml.sax.saxutils import unescape
import ftfy
from unidecode import unidecode
import io
import editdistance
import os 
import fnmatch

# ACL Anthology Network
AAN_DIR = '../resources/aan/2014/'

# ACL Converted to JSON
PATH    = '../data/arc-parsit-dir/'

def clean(title):
    if title[-4:] == 'quot':
        title = title[0:-4]
    if title[-3:] == ' in':
        title = title[0:-3]
    if title[-5:] == ' corr':
        title = title[0:-5]

    title = re.sub(r'[^a-zA-Z0-9]+', ' ', title)
    title = re.sub(r'[ ]+', ' ', title)
    title = re.sub(r'^[0-9\-]+[a-f]?', '', title)
    
    return title

html_parser = HTMLParser()

exclude = set(string.punctuation)

def cleanTitle(title):
    title = title.lower().strip()
    title = ''.join(ch for ch in title if ch not in exclude)
    return clean(title)

arc_only_network = defaultdict(list)
with open(AAN_DIR + 'networks/paper_citation_network.txt') as f:
    for line in f:
        cols = line.strip().split(" ==> ")
        arc_only_network[cols[0]].append(cols[1])

author_id_to_name = {}
def load_author_data():
    with io.open(AAN_DIR + 'author_ids.txt', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            auth_id = cols[0]
            name = cols[1]
            name = name.replace('&cacute;', 'c') \
                .replace('&scedil;', 's') \
                .replace('&Scedil;', 'S') \
                .replace('&Ccaron;', 'C') \
                .replace('&ccaron;', 'c') \
                .replace('&scaron;', 's') \
                .replace('&#scaron;', 's') \
                .replace('&ncaron;', 'n') \
                .replace('&zcaron;', 'z') \
                .replace('&slig;', 'ss') \
                .replace('&nacute;', 'n') \
                .replace('&Atilde;', 'A') \
                .replace('&Sacute;', 'S') \
        
            name = html_parser.unescape(unescape(name))
        #name = ftfy.fix_text(name.encode('utf-8'))
        #name = name.encode('utf-8')
            name = unidecode(name)
            if '&' in name:
            #print cols[1], '->',  name.encode('utf-8')
                pass

            names = name.split(',')
            name = (names[1] + ' ' + names[0]).strip()

            author_id_to_name[auth_id] = name



arc_id_to_paper = {}

def load_paper_data():
    
    load_author_data()

    paper_id_to_author_ids = defaultdict(set)
    with open(AAN_DIR + '/paper_author_affiliations.txt') as f:
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 2:
                continue
            paper_id_to_author_ids[cols[0]].add(cols[1])

    print 'Loaded author IDs for %d papers' % (len(paper_id_to_author_ids))
            
    
    with open(AAN_DIR + '/paper_ids.txt') as f:
        lines = f.readlines()
        skip_next = False
        for i, line in enumerate(lines):

            if skip_next:
                skip_next = False
                continue

            cols = line.strip().split('\t')

            if len(cols) < 3:
                line = line + '\t' + lines[i+1].strip()
                cols = line.split('\t')
                skip_next = True
            elif len(cols) > 3:
                print len(cols), line

            title = cols[1]
            title = cleanTitle(title)
            year = cols[2]
            paper_id = cols[0]
            author_ids = paper_id_to_author_ids[paper_id]

            try:
                int(year)
            except:
                print 'Bad year for line' + line.strip()
                continue

            author_names = []
            for aid in author_ids:
                author_names.append(author_id_to_name[aid])

            # print '%s -> %s -> %s' % (paper_id, str(author_ids), str(author_names))
        
            arc_id_to_paper[paper_id] = (title, year, author_names)
            
    with io.open('../resources/paper_author_title_year.extra.tsv', encoding='utf-8') as f:
        extras = 0
        for line in f:
        
            cols = line.strip().split('\t')
            if len(cols) != 3:
            #print 'bad line?',  line
                continue
            arc_id = cols[0]
            if arc_id not in arc_id_to_paper:
                arc_id_to_paper[arc_id] = (cols[1], cols[2], [])            
                extras += 1

    print 'Added %d extra IDs' % (extras)


def load_paper_data_old():
    with open("../resources/paper_author_title_year.csv","rb") as f:
        rdr = unicodecsv.reader(f, encoding='ISO-8859-1')
        for entry in rdr:
            title = entry[-2]
            title = cleanTitle(title)
            year = entry[-1]
            paper_id = entry[0]
            author_ids = entry[1:-2]

            author_names = []
            for aid in author_ids:
                author_names.append(author_id_to_name[aid])
        
            arc_id_to_paper[paper_id] = (title, year, author_names)

    with io.open('../resources/paper_author_title_year.extra.tsv', encoding='utf-8') as f:
        extras = 0
        for line in f:
        
            cols = line.strip().split('\t')
            if len(cols) != 3:
            #print 'bad line?',  line
                continue
            arc_id = cols[0]
            if arc_id not in arc_id_to_paper:
                arc_id_to_paper[arc_id] = (cols[1], cols[2], [])            
                extras += 1

    print 'Added %d extra IDs' % (extras)


EXTERNAL_PAPER_IDS = 0

def getAuthors(title):
    return paper_data[title]["authors"]

def getYear(title):
    return int(paper_data[title]["year"])


def start_overlap(s1, s2):
    s1 = s1.replace(' ', '')
    s2 = s2.replace(' ', '')
    n = 0.0
    for i, c in enumerate(s1):
        if i == len(s2) or c != s2[i]:
            break
        n += 1
    l1 = 0
    if len(s1) > 0:
        l1 = n / len(s1)
    l2 = 0
    if len(s2) > 0:
        l2 = n / len(s2)
    return (l1, l2)


def find_match(arc_paper_ids, title, year, authors):
    for pid in arc_paper_ids:
        if pid not in arc_id_to_paper:
            continue
        paper_data = arc_id_to_paper[pid]
        
        #print 'checking "%s" and "%s"' % (title, paper_data[0])

        if title in paper_data[0] or paper_data[0] in title:
            #print 'Saw match between "%s" and "%s"' % (title, paper_data[0])
            return pid

    return None


def authors_match(authors1, authors2):
    #print '%s ?= %s' % (str(authors1), str(authors2))

    if len(authors1) != len(authors2):
        return False
    
    names1 = set()
    names2 = set()
    
    for a1 in authors1:
        n1 = a1.lower().replace('-', '').split()
        names1.add(n1[-1])
    for a2 in authors2:
        n2 = a2.lower().replace('-', '').split()
        names2.add(n2[-1])

    if names1 == names2:
        return True

    else:
        return authors_match2(authors1, authors2)

def authors_match2(authors1, authors2):
    #print '%s ?= %s' % (str(authors1), str(authors2))

    if not ((len(authors1) > 3 and abs(len(authors1) - len(authors2)) < 2) \
                or len(authors1) == len(authors2)):
        print 'Not equal %d, %d  %s != %s' % (len(authors1), len(authors2), str(authors1), str(authors2))
        return False

    if len(authors2) > len(authors1):
        tmp = authors1
        authors1 = authors2
        authors2 = tmp

    
    num_mismatches = 0
    k = 0

    for i, a1 in enumerate(authors1):
        if k > 0:
            if i != len(authors2) + k:
                num_mismatches += 1
            continue
        elif i >= len(authors2):
            num_mismatches += 1
            continue
        
        
        a1 = authors1[i+k]
        a2 = authors2[i]

        n1 = a1.lower().replace('-', '').split()
        n2 = a2.lower().replace('-', '').split()

        if eq(n1, n2):
            continue
        else:
            #return False
            num_mismatches += 1
            if len(authors1) > len(authors2) and eq(authors1[i+1].lower().replace('-','').split(), n2):
                k += 1
                #print 'Saw next-author match'
            else:
                #print ('not equal: "%s" != "%s"  (%s :: %s)' % (a1, a2, authors1, authors2)).encode('utf-8')
                pass
                
    #print 'num_mismatches: ', num_mismatches
    return num_mismatches == 0 or (len(authors1) > 3 and num_mismatches <= 1)

def eq(n1, n2):
    if n1[-1] == n2[-1]:
        return True
        # Check if names got swapped
    elif n1[-1] == n2[0] or n1[0] == n2[-1]:
        return True
    elif (len(n2) >= 2 and n1[-1] == n2[-2]) or (len(n1) >= 2 and n1[-2] == n2[-1]):
        return True
    elif editdistance.eval(n1[-1], n2[-1]) == 1:
            #print ('EDIT-equal: "%s" ~= "%s"  (%s :: %s)' % (a1, a2, authors1, authors2)).encode('utf-8')
        return True
    else:
        return False

def expand(group, unr_to_matches):
    while True:
        new = set()
        for u in group:
            for u2 in unr_to_matches[u]:
                if u2 not in group:
                    new.add(u2)
        if len(new) == 0:
            break
        for u in new:
            group.add(u)

            
def main(): 
    if len(sys.argv) != 3:
        print 'Usage: canonicalize_citations.py out-data.tsv out-net.tsv'
        return

    load_paper_data()

    output_data_file = sys.argv[1]
    output_net_file = sys.argv[2]

    #citations = defaultdict(list)
    citations_to_resolve = defaultdict(list)
    unresolved_ids = {}

    num_processed = 0

    matched_citations = []

    files = []
    for root, dirnames, filenames in os.walk(PATH):
        for filename in fnmatch.filter(filenames, '*.xml'):
            files.append(os.path.join(root, filename))

    print 'Processing %d files' % len(files)

    for fi, f in enumerate(files):
        #if 'S15-2017' not in basename(f):
        #    continue;
        #if fi > 500:
        #    break
        try:
            xmldoc = minidom.parse(f)
        except ExpatError:
            # print 'Bad file?: %s' % (f)
            continue
                
        paper_id = basename(f)[:-4]
        #print paperId
        paper_id = paper_id[0:8]

        known_arc_citations = []
        #print paper_id
        if paper_id in arc_only_network:
            known_arc_citations = arc_only_network[paper_id]
            #print '%s -> %s' % (paper_id, known_arc_citations)

        itemlist = xmldoc.getElementsByTagName('citation')
        for s in itemlist:
            try:
                title_elem = s.getElementsByTagName("title")
                    # Need a title
                if title_elem is None or len(title_elem) == 0 or title_elem[0].firstChild is None:
                    continue

                title = cleanTitle(title_elem[0].firstChild.data)
                date_elem = s.getElementsByTagName("date")
                    # Need a year
                if date_elem is None or len(date_elem) == 0 or date_elem[0].firstChild is None:
                    continue
                year = date_elem[0].firstChild.data

                authors = []
                clean_authors = []
                for elem in s.getElementsByTagName("author"):
                    name = elem.firstChild.data
                    authors.append(name)
                    name = unidecode(name)
                    name = name.replace("'", '').replace('~','')\
                        .replace(",",'').replace('"', '').replace('`', '')
                    clean_authors.append(name)
                    
                # print ('saw cited paper %s (%s) "%s"' % (', '.join(authors), year, title)).encode('utf-8')
                
                # See if this citation is an in-ARC citation from the known
                # citations
                clean_title = clean(title)
                arc_id = find_match(known_arc_citations, clean_title, year, authors)
                if arc_id is None:
                    ex_id = 'Unresolved_%d' % (len(unresolved_ids))
                    unresolved_ids[ex_id] = {'title':title, 'year': year, \
                                                 'authors': authors, 'clean_title': clean_title, \
                                             'clean_authors': clean_authors}                
                    citations_to_resolve[paper_id].append(ex_id)
                else:
                    matched_citations.append((arc_id, title, year, authors))
                    
            except IndexError, KeyError:
                continue

            contexts = s.getElementsByTagName("context")
            
            relTxt = ""
            for context in contexts:
                citeStr = context.getAttribute("citStr")
                citeTxt = context.firstChild.data

        num_processed += 1
        if num_processed % 1000 == 0:
            print 'Processed %d files' % (num_processed)
                #break

    print 'Found %d citations in the ARC' % (len(matched_citations))


    unr_to_matches = defaultdict(list)

    print 'Assigning prefices'

    prefix_to_unresolved_ids = defaultdict(dict)
    for uid, data in unresolved_ids.iteritems():
        title = data['clean_title']
        prefix = title[0:min(3,len(title))]
        subset = prefix_to_unresolved_ids[prefix]
        subset[uid] = data

    print 'Canonicalizing %d unresolved citations' % (len(unresolved_ids))

    # Now try to match
    u_count = 0
    for uid, data in unresolved_ids.iteritems():

        u_count += 1
        if u_count % 1000 == 0:
            print 'Processing unresolved item %d / %d' % (u_count, len(unresolved_ids))

        title1 = data['clean_title']
        year = int(data['year'])

        possible_arc_matches = []

        for arc_id, arc_data in arc_id_to_paper.iteritems():
            title2 = arc_data[0]
            m1, m2 = start_overlap(title1, title2)
            if m1 > 0.9 or m2 > 0.9:
                #print 'ARC MATCH: %f : %f -> "%s" (%s) ==  "%s" (%s)' % (m1, m2, title1, data['year'], title2, arc_id)
                if authors_match(data['clean_authors'], arc_data[2]):
                    possible_arc_matches.append(arc_id)
                else:
                    #print 'No author match!'
                    pass

        
        closest_year_diff = 1000
        for arc_id in possible_arc_matches:
            arc_year = int(arc_id_to_paper[arc_id][1])
            if year == arc_year:
                data['id'] = arc_id
                # print '%s --> %s' % (uid, arc_id)
                break
            else:
                year_diff = year - arc_year
                if abs(year_diff) < closest_year_diff:
                    closest_id = arc_id
                    closest_year_diff = abs(year_diff)

        if closest_year_diff <= 1:
            # print '%s -> %s' % (uid, closest_id)
            data['id'] = closest_id

        # We found a match for this in the ARC
        if 'id' in data:
            continue

        prefix = title1[0:min(3,len(title1))]
        unresolved_ids_subset = prefix_to_unresolved_ids[prefix]

        for uid2, data2 in unresolved_ids_subset.iteritems():
            if uid == uid2:
                break

            # Skip pairs with bad author matches
            if len(data['clean_authors']) != len(data2['authors']):
                continue
            
            title2 = data2['clean_title']

            #if data['clean_title'].startswith(data2['clean_title']) \
            #        or data2['clean_title'].startswith(data['clean_title']):
            
            m1, m2 = start_overlap(title1, title2)
            if m1 > 0.9 or m2 > 0.9:
                if authors_match(data['clean_authors'], data2['clean_authors']):
                    unr_to_matches[uid].append(uid2)
                    unr_to_matches[uid2].append(uid)
                    #print 'EXT_MATCH %f : %f -> "%s"  ==  "%s"' % (m1, m2, title1, title2)

    print 'Resolving'
                
    # Group all the unresolved papers together to figure out what the right
    # information is
    already_resolved = set()
    num_external_ids = 0
    num_internal_ids = 0
    for uid, data in unresolved_ids.iteritems():
        if uid in already_resolved:
            continue
        if 'id' in data:
            num_internal_ids += 1
            continue
        group = set()
        group.add(uid)
        expand(group, unr_to_matches)

        #print 'Assigning to same ID:'
        #for u in group:
        #    print '\t%s (%s)' % (unresolved_ids[u]['title'], unresolved_ids[u]['year'])
        #print ''
        ex_id = "External_%d" % (num_external_ids)

        num_external_ids += 1
        for u in group:
            unresolved_ids[u]['id'] = ex_id
            already_resolved.add(u)

    print 'Resolved %d external papers to %d ids (%d external)'  \
        % (len(unresolved_ids), num_external_ids + num_internal_ids, num_external_ids)

    with open(output_net_file, 'w') as n_out:
        with open(AAN_DIR + 'networks/paper_citation_network.txt') as f:
            for line in f:
                cols = line.strip().split(" ==> ")
                n_out.write('%s\t%s\n' % (cols[0], cols[1]))
                       
        for citing_id, uids in citations_to_resolve.iteritems():
            for uid in uids:
                n_out.write('%s\t%s\n' % (citing_id, unresolved_ids[uid]['id']))
    
            

    with open(output_data_file, 'w') as d_out:

        for arc_id, title, year, authors in matched_citations:
            d_out.write(('%s\t%s\t%s\t%s\n' % (arc_id, year, title, ', '.join(authors))).encode('utf-8'))

        for arc_id, (title, year, authors) in arc_id_to_paper.iteritems():
            d_out.write(('%s\t%s\t%s\t%s\n' % (arc_id, year, title, ', '.join(authors))).encode('utf-8'))

        for uid, data in unresolved_ids.iteritems():
            ex_id = data['id']
            year = data['year']
            title = data['title']
            authors = data['authors']
            d_out.write(('%s\t%s\t%s\t%s\n' % (ex_id, year, title, ', '.join(authors))).encode('utf-8'))
            

if __name__ == "__main__":
    main()
