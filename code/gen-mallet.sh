#!/bin/bash
# Strip out the non-ASCII stuff that Mallet chokes on and remove the
# citation-related stopwords

MALLET_HOME=/shared/0/resources/mallet-2.0.8/bin/

base_dir=../working-dir/topics/

echo 'cleaning context'
# perl -pi -e 's/[[:^ascii:]]//g' < $base_dir/citance-contexts.txt | sed -e 's/ [0-9][0-9][0-9][0-9] / /g' | sed -e 's/-lrb-\|-rrb-\|-lsb-\|-rsb-\|,\|\.//g' | sed -e 's/ al. \| al \| et / /g' > $base_dir/citance-contexts.cleaned.txt

echo 'cleaning extended context'
perl -pi -e 's/[[:^ascii:]]//g' < $base_dir/extended-citance-contexts.txt \
    | sed -e 's/ [0-9][0-9][0-9][0-9] / /g' \
    | sed -e 's/-lrb-\|-rrb-\|-lsb-\|-rsb-\|,\|\.//g' \
    | sed -e 's/ al. \| al \| et / /g' \
          > $base_dir/extended-citance-contexts.cleaned.txt

# Remove rare tokens
echo 'removing rare words from extended citance contexts'
python remove_rare_words.py $base_dir/extended-citance-contexts.cleaned.txt $base_dir/extended-citance-contexts.cleaned.trimmed.txt
echo 'removing rare words from citance contexts'
python remove_rare_words.py $base_dir/citance-contexts.cleaned.txt $base_dir/citance-contexts.cleaned.trimmed.txt



# Generate the mallet stuff
echo 'converting context to mallet file'
$MALLET_HOME/mallet import-file --preserve-case --keep-sequence --remove-stopwords --token-regex '\S+' --input $base_dir/citance-contexts.cleaned.trimmed.txt --output $base_dir/citance-contexts.mallet

echo 'converting extended context to mallet file'
$MALLET_HOME/mallet import-file --preserve-case --keep-sequence --remove-stopwords --token-regex '\S+' --input $base_dir/extended-citance-contexts.cleaned.trimmed.txt --output $base_dir/extended-citance-contexts.mallet
