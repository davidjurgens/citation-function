#!/bin/bash

#  #../working-dir/arc-ftrs/
if [ -z "$1" ]; then
   feature_vector_dir=../working-dir/arc-ftrs/
else
    feature_vector_dir=$1
fi

if [ -z "$2" ]; then
    base_output_dir=../results/
else
    base_output_dir=$2
fi

if [ -z "$3" ]; then
    classifier=../working-dir/function-classifier.pkl
else
    classifier=$3
fi

ct_output_dir=$base_output_dir/cite-func

if [ ! -d "$ct_output_dir" ] ; then
    mkdir -p $ct_output_dir
fi



ls $feature_vector_dir/ | sed -e 's/.ftr//g' | shuf  \
    | xargs --verbose -P 80 -n 1 -I % python classify_papers.py \
            $classifier \
            $feature_vector_dir/%.ftr \
            $ct_output_dir/%.tsv 
    
find $ct_output_dir/ -name '*.tsv' | xargs cat > $base_output_dir/cite-func.all.tsv
