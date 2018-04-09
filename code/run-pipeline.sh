

# Download the ARC

# Canonicalize all the outgoing citations in the ARC, which resolves external
# IDs to a single entity
#python canonicalize_citations.py \
#    ../working-files/arc-citation-network.csv \
#    ../working-files/arc-paper-ids.tsv

# Convert the XML formatted text to JSON, with resolved citation contexts
#python convert_ARC_xml_to_json.py ../data/arc-json

# Convert Teufel's data to the JSON format
#python convert_teufel_data.py ../data/teufel-json

# Integrate the BRAT annotations into the json data
#python integrate_annotations.py ../data/annotated-json-data

# Generates a corpus of citances and citance IDs to feed into Mallet for topic
# modeling
python generate_mallet_corpus.py ../working-files/

./train-mallet-models.sh

# Weight each cited paper based on its centrality, PageRank, etc. in the
# citation network
# python compute_temporal_weights.py ../working-files/arc-paper-ids.2.tsv ../working-files/arc-network-weights.tsv

# Generate the training data
#python generate_training_data.py 

# Use the features found by the training data to covert the ARC into feature vectors
#python convert_ARC_to_features.py

# Classify everything 
#python classify_papers.py

