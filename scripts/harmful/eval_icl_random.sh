#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}

EVAL_DOMAINS=(
    "discrimination,stereotype,injustice" 
    "hate_speech,offensive_language" 
    "non_violent_unethical_behavior"
)
#     "violence,aiding_and_abetting,incitement"
# )

ROOT="/mnt/bn/domain-adaptation/mlx/users/hongyi.guo/repo/5923/self-alignment"
MODEL_PATH=$ROOT/output/$MODEL/pretrain/model

for DOMAIN in ${EVAL_DOMAINS[@]}; do
    for NUM in 64 128; do
        DIR="$ROOT/output/$MODEL/icl_random/$NUM/$DOMAIN"
        mkdir -p $DIR
        echo "Evaluating $MODEL on domain $DOMAIN"
        echo "Writing to $DIR/output.log"

        if [ ! -f "$DIR/summary.json" ]; then
            python3 self_alignment/generate.py \
            --model_name_or_path $MODEL_PATH \
            --batch_size 2 \
            --random \
            --eval_path $ROOT/data/$DOMAIN/64/eval.json \
            --dataset_paths $ROOT/data/$DOMAIN/$NUM/train.json \
            --output_dir $DIR \
            --evaluate \
            &> $DIR/output.log
        fi
    done
done
