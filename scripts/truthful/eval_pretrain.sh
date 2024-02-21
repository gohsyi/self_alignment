#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}

ROOT="/mnt/bn/domain-adaptation/mlx/users/hongyi.guo/repo/5923/self-alignment"
MODEL_PATH=$ROOT/output/$MODEL/pretrain/model

DIR="$ROOT/output/$MODEL/pretrain/truthful_qa"
mkdir -p $DIR
echo "Writing to $DIR/output.log"
python3 self_alignment/generate.py \
--batch_size 2 \
--model_name_or_path $MODEL_PATH \
--eval_path $ROOT/data/truthful_qa/eval.json \
--output_dir $DIR \
&> $DIR/output.log

python3 -m truthfulqa.evaluate --input_path $DIR/eval.json
