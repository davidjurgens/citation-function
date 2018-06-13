#from snap import *
import sys
from collections import *
from networkx import *

def main():

    if len(sys.argv) != 3:
        print 'usage: input.graph output.tsv'
        return

    outfile = sys.argv[2]

    paper_id_to_year = {}

    paper_id_to_authors = {}
    author_to_paper_ids = defaultdict(list)    
    author_id_to_name = {}
    name_to_author_ids = defaultdict(list)
    last_name_to_ids = defaultdict(list)
    abbreviated_counts = Counter()


    print 'Loading paper author info'
    with open("../resources/arc-paper-ids.tsv", "rb") as f:
        for line in f:
            cols = line.strip().split('\t')
            # Happens if there were no authors for the paper (as determined by
            # ParsCit)
            if len(cols) != 4:
                #print 'bad line?: ' + line.strip()
                continue
            paper_id = cols[0][0:8]
            year = int(cols[1])
            authors = cols[3].split(', ')

            paper_id_to_year[paper_id] = year
                       
            # Make an effort to try canonicalizing the names 
            if paper_id.startswith('ARC_External'):
                for i, name in enumerate(authors):
                    inferred_id = find_id(name, last_name_to_ids, \
                                          author_id_to_name, abbreviated_counts)
                    if inferred_id is not None:
                        authors[i] = inferred_id

            paper_id_to_authors[paper_id] = authors
            for a in authors:
                author_to_paper_ids[a].append(paper_id)


    graph = defaultdict(list)
    edges = []
    paper_id_to_vertex = {}
    vertex_to_paper_id = {}
    vertex_to_year = {}

    with open(sys.argv[1]) as gf:
        for line in gf:

            cols = line.strip().split('\t')
            citing = cols[0][0:8]
            cited = cols[1]

            #if 'External' in line:
            #    continue

            try:
                year = int(citing[1:3])
            except BaseException as e:
                print citing
                #raise e
                continue
            if year < 50:
                year += 2000
            else:
                year += 1900

            graph[citing].append(cited)
            if citing not in paper_id_to_vertex:
                vertex = len(paper_id_to_vertex)
                paper_id_to_vertex[citing] = vertex
                vertex_to_paper_id[vertex] = citing
                vertex_to_year[paper_id_to_vertex[citing]] = year
            elif  paper_id_to_vertex[citing] not in vertex_to_year:
                vertex_to_year[paper_id_to_vertex[citing]] = year

            if cited not in paper_id_to_vertex:
                vertex = len(paper_id_to_vertex)
                paper_id_to_vertex[cited] = vertex
                vertex_to_paper_id[vertex] = cited

            citing_v = paper_id_to_vertex[citing]
            vertex_to_year[citing_v]
            cited_v = paper_id_to_vertex[cited]

            edges.append((citing_v, cited_v, year))

            #    for i, paper_id in enumerate(paper_id_to_year):
            #        paper_id_to_vertex[paper_id] = i
            #        vertex_to_paper_id[i] = paper_id


    with open(outfile, 'w') as outf:
        outf.write('\t'.join(['year', 'arc_id', 'pagerank', 'in_degree', 'hub', 'authority', 'load_centrality', ]) + '\n')
        for max_year in range(1979, 2016):
            g = get_network(edges, vertex_to_year, max_year)

            print 'Build network from %d and earlier from %d vertices and %d edges' \
                % (max_year, g.order(), g.size())

            print '  computing page rank'
            pageranks = nx.pagerank(g, alpha=0.85)
            print '  computing HITS'
            hub_scores, auth_scores = nx.hits_scipy(g, max_iter=1000)
            print '  computing centrality'
            load_centralities = nx.load_centrality(g)

            print '  writing results'
            for v, pr in pageranks.iteritems():
                arc_id = vertex_to_paper_id[v]
                in_degree = g.in_degree(v) # num of citations
                hub = hub_scores[v]
                auth = auth_scores[v]
                load_c = load_centralities[v]
                outf.write('\t'.join([str(x) for x in [max_year, arc_id, pr, in_degree, 
                                                       hub, auth, load_c]]) + '\n')
                                                       
                

def get_network(edges, vertex_to_year, max_year):
    g = nx.DiGraph()        
    for edge in edges:
        if edge[2] > max_year:
            continue
        g.add_edge(edge[0], edge[1])
    return g
    

def find_id(external_name, last_name_to_ids, author_id_to_name, abbreviated_counts):
    name_parts = external_name.split()
    last_name = name_parts[-1]

    # Handle cases where the first name's initial and middle initial are
    # erroneously combined into a single word, e.g.,     
    # "C. D. Manning" -> "CD Manning"

    # NOTE: we see some weirdness with Asian name formats where the last name is
    # captialized and put either first or last (can't tell why), e.g., "HASHIDA
    # Koiti", which means we should probably re-order the name.  However(!),
    # sometimes all the words are capitalized
    tmp = []
    for i, n in enumerate(name_parts):
        if len(n) == 1:
            tmp.append(n)
        # Treat this as two collapsed initials 
        elif len(n) == 2 and n == n.upper():
            tmp.append(n[0])
            tmp.append(n[1])
        elif len(n) >= 3 and n == n.upper():
            tmp.append(n[0] + n[1:].lower())


    # Cases to check...
    #
    # Official: Manning, Christopher D.
    #
    # C D Manning
    # C Manning
    # CD Manning
    
    # See which authors with the same last name might have the same first name
    authors_with_same_last_name = last_name_to_ids[last_name]
    
    # NOTE: sometimes people with hyphenated last names get their names mangled
    # by ParsCit. Maybe try fixing these? (rare case though)
   

    for author_id in authors_with_same_last_name:

        # Get the forenames of the AAN author
        aan_name = author_id_to_name[author_id]
        aan_forenames = aan_name.split(',')[1].replace('.', ' ').split()
        
        found = True
        for i, name in enumerate(aan_forenames):
            if i + 1 >= len(name_parts):
                # If this isn't the first name and this is a middle initial and
                # there are no more names to match for this AAN name
                if i > 0 and len(name) == 1 and i + 1 == len(aan_forenames):
                    # See if this is an unambiguous name if we match just on the
                    # first initial and last name
                    if abbreviated_counts[name + ' ' + last_name] > 1:
                        found = False
                        break
                else:
                    found = False
                    break
            # It could be that the External name is missing a middle initial
            elif not name.startswith(name_parts[i]):
                found = False
                break
        
        if found:
            #print '%s ?= %s :: %s' % (external_name, aan_name, str(found))
            return author_id
        

    return None
    



if __name__ == "__main__":
    main()
