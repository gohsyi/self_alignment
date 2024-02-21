#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}
SAMPLES=${2:-64}
SAMPLES_GEN=${3:-512}
SRC=${4:-"truthful_qa"}
EPOCHS=${5:-5}
PEER=${6:-0}

ROOT="/mnt/bn/domain-adaptation/mlx/users/hongyi.guo/repo/5923/self-alignment"
DIR="$ROOT/output/$MODEL/self_alignment_v0_peer$PEER/$SRC/${SAMPLES}_${SAMPLES_GEN}"

mkdir -p $DIR/0/data

# cp -r $ROOT/output/$MODEL/pretrain/$MODEL $DIR/0/sft
cp $ROOT/data/$SRC/$SAMPLES/train.json $DIR/0/data/train.json

PRETRAIN_PATH=$ROOT/output/$MODEL/pretrain/model
MODEL_PATH=$PRETRAIN_PATH

# Learning rate is 2e-5
LEARNING_RATE=0.00002

for EPOCH in $(eval echo "{1..$EPOCHS}"); do
    echo "******* epoch $EPOCH *******"
    DATA_DIR=$DIR/$EPOCH/data
    SFT_DIR=$DIR/$EPOCH/sft
    EVAL_DIR=$DIR/$EPOCH/eval

    echo "Generating dataset"
    echo "Writing to $DATA_DIR/output.log"
    mkdir -p $DATA_DIR
    
    if [ ! -f "$DATA_DIR/train.json" ]; then
        python3 self_alignment/generate.py \
        --model_name_or_path $MODEL_PATH \
        --generate $SAMPLES_GEN \
        --dataset_paths $(eval echo "$DIR/{0..$(( EPOCH - 1))}/data/train.json") \
        --do_sample \
        --batch_size 3 \
        --verbose \
        --threshold_rouge 0.7 \
        --threshold_length 5 \
        --filter \
        --output_dir $DATA_DIR &> $DATA_DIR/output.log
    fi

    echo "Training with learning rate $LEARNING_RATE"
    echo "Writing to $SFT_DIR/output.log"
    mkdir -p $SFT_DIR

    if [ ! -f "$SFT_DIR/pytorch_model.bin" ]; then
        bash scripts/sft.sh \
        --model_name_or_path $MODEL_PATH \
        --datasets "local_json:1.0:$DIR/0/data/train.json&$DATA_DIR/train.json" \
        --learning_rate $LEARNING_RATE \
        --zero_stage 2 \
        --offload all \
        --peer $PEER \
        --output_dir $SFT_DIR &> $SFT_DIR/output.log
    fi

    LEARNING_RATE=$(awk "BEGIN {printf \"%.10f\",${LEARNING_RATE}/2}")

    echo "Evaluating"
    mkdir -p $EVAL_DIR/$SRC

    if [ ! -f $EVAL_DIR/$SRC/eval.json ]; then
        cp $ROOT/data/$SRC/eval.json $EVAL_DIR/$SRC/eval.json

        echo "Writing to $EVAL_DIR/$SRC/output.log"
        python3 self_alignment/generate.py \
        --model_name_or_path $SFT_DIR \
        --eval_path $ROOT/data/$SRC/eval.json \
        --output_dir $EVAL_DIR/$SRC &> $EVAL_DIR/$SRC/output.log
    fi

    if [ ! -f $EVAL_DIR/$SRC/summary.csv ]; then
        python3 -m truthfulqa.evaluate --input_path $EVAL_DIR/$SRC/eval.json
    fi
    
    MODEL_PATH=$SFT_DIR

done
