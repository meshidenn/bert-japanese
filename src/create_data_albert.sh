#!/bin/bash

set -eu

basedir=$1
max_seq_length=$2

for DIR in $( find ${basedir}/data/ -mindepth 1 -type d ); do 
  python3 src/create_pretraining_data.py \
    --input_file=${DIR}/all.txt \
    --output_file=${DIR}/all-maxseq${max_seq_length}.tfrecord \
    --model_file=${basedir}/model/wiki-ja.model \
    --vocab_file=${basedir}/model/wiki-ja.vocab \
    --do_lower_case=True \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=40 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5 \
    --do_whole_word_mask=False
done
