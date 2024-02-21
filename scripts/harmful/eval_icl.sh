#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}

EVAL_DOMAINS=(
    "discrimination,stereotype,injustice" 
    "hate_speech,offensive_language" 
    "non_violent_unethical_behavior"
)

# MODEL_PATH=$ROOT/output/$MODEL/pretrain/model
MODEL_PATH="huggyllama/llama-7b"

for DOMAIN in ${EVAL_DOMAINS[@]}; do
    for EVAL_DOMAIN in ${EVAL_DOMAINS[@]}; do
        NUM="64"
        DIR="$ROOT/output/$MODEL/icl/$NUM/$DOMAIN/$EVAL_DOMAIN"
        mkdir -p $DIR
        echo "Evaluating $MODEL on domain $DOMAIN"
        echo "Writing to $DIR/output.log"

        if [ ! -f "$DIR/summary.json" ]; then
            python3 self_alignment/generate.py \
            --model_name_or_path $MODEL_PATH \
            --batch_size 2 \
            --eval_path data/$EVAL_DOMAIN/64/eval.json \
            --dataset_paths data/$DOMAIN/$NUM/train.json \
            --output_dir $DIR \
            --evaluate \
            &> $DIR/output.log
        fi
    done
done
