#!/bin/bash

# FILL IN
MALLET_HOME=

#for num_topics in `echo 1000` ; do
for num_topics in `echo 100 ` ; do

    output_dir=../working-files/topics3/$num_topics-topics/
    if [ ! -d "$output_dir" ] ; then
        mkdir -p $output_dir
    fi
    
    $MALLET_HOME/bin/mallet train-topics \
        --input $output_dir/extended-citance-contexts.mallet \
        --output-topic-keys $output_dir/extended-citance.topics-words.txt \
        --num-top-words 100 \
        --output-doc-topics $output_dir/extended-citance.doc-topics.txt \
        --inferencer-filename $output_dir/extended-citance.inferencer.mallet \
        --num-topics $num_topics \
        --num-threads 64 \
        --num-iterations 2000 \
       --optimize-interval 100 &

    $MALLET_HOME/bin/mallet train-topics \
        --input $output_dir/extended-citance-contexts.mallet \
        --output-topic-keys $output_dir/citance.topics-words.txt \
        --num-top-words 100 \
        --output-doc-topics $output_dir/citance.doc-topics.txt \
        --inferencer-filename $output_dir/citance.inferencer.mallet \
        --num-topics $num_topics \
        --num-threads 64 \
        --num-iterations 2000 \
        --optimize-interval 100

    wait
done
