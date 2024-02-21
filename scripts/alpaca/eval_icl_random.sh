#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}

ROOT="/mnt/bn/domain-adaptation/mlx/users/hongyi.guo/repo/5923/self-alignment"
MODEL_PATH=$ROOT/output/$MODEL/pretrain/model

# for NUM in 64 128 256 512; do

NUM=64

DIR="$ROOT/output/$MODEL/icl_random/$NUM/alpaca"
mkdir -p $DIR
echo "Writing to $DIR/output.log"
python3 self_alignment/generate.py \
--model_name_or_path $MODEL_PATH \
--context_nums 4 \
--batch_size 2 \
--random \
--eval_path $ROOT/data/alpaca/eval.json \
--dataset_paths $ROOT/data/alpaca/$NUM/train.json \
--output_dir $DIR \
&> $DIR/output.log

# python3 -m truthfulqa.evaluate --input_path $DIR/eval.json

# done
