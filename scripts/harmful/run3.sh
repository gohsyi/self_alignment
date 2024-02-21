#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}
SAMPLES=${2:-64}
SAMPLES_GEN=${3:-512}
SRC=${4:-"mix"}
EPOCHS=${5:-5}
PEER=${6:-0}

EVAL_DOMAINS=(
    "discrimination,stereotype,injustice"
    "hate_speech,offensive_language"
    "non_violent_unethical_behavior"
    "violence,aiding_and_abetting,incitement"
)

ROOT="/mnt/bn/domain-adaptation/mlx/users/hongyi.guo/repo/5923/self-alignment"
DIR="$ROOT/output/$MODEL/self_alignment_v3_peer$PEER/$SRC/${SAMPLES}_${SAMPLES_GEN}"

mkdir -p $DIR/0/data

PRETRAIN_PATH=$ROOT/output/$MODEL/pretrain/model
MODEL_PATH=$PRETRAIN_PATH

cp $ROOT/data/$SRC/$SAMPLES/train.json $DIR/0/data/train.json

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
    
    # Question generation
    python3 self_alignment/generate/inputs.py \
    --model_name_or_path $PRETRAIN_PATH \
    --generate $SAMPLES_GEN \
    --dataset_paths $(eval echo "$DIR/{0..$(( EPOCH - 1))}/data/train.json") \
    --batch_size 4 \
    --threshold_rouge 0.7 \
    --do_sample \
    --filter \
    --save_name train.json \
    --output_dir $DATA_DIR &> $DATA_DIR/output.log

    # Answer generation
    python3 self_alignment/generate/answers.py \
    --model_name_or_path $MODEL_PATH \
    --eval_path $DATA_DIR/train.json \
    --dataset_paths $(eval echo "$DIR/{0..$(( EPOCH - 1))}/data/train.json") \
    --do_sample \
    --batch_size 4 \
    --threshold_length 5 \
    --filter \
    --save_name train.json \
    --output_dir $DATA_DIR &> $DATA_DIR/output.log

    echo "Training with learning rate $LEARNING_RATE"
    echo "Writing to $SFT_DIR/output.log"
    mkdir -p $SFT_DIR

    bash scripts/sft.sh \
    --model_name_or_path $MODEL_PATH \
    --datasets "local_json:1.0:$DIR/0/data/train.json&$DATA_DIR/train.json" \
    --learning_rate $LEARNING_RATE \
    --zero_stage 2 \
    --offload all \
    --peer $PEER \
    --output_dir $SFT_DIR &> $SFT_DIR/output.log


    LEARNING_RATE=$(awk "BEGIN {printf \"%.10f\",${LEARNING_RATE}/2}")

    echo "Evaluating"
    for DOMAIN in ${EVAL_DOMAINS[@]}; do
        mkdir -p $EVAL_DIR/$DOMAIN
        cp $ROOT/data/$DOMAIN/$SAMPLES/eval.json $EVAL_DIR/$DOMAIN/eval.json

        echo "Writing to $EVAL_DIR/$DOMAIN/output.log"
        python3 self_alignment/generate.py \
        --model_name_or_path $SFT_DIR \
        --eval_path $ROOT/data/$DOMAIN/$SAMPLES/eval.json \
        --evaluate \
        --output_dir $EVAL_DIR/$DOMAIN &> $EVAL_DIR/$DOMAIN/output.log
    done

    MODEL_PATH=$SFT_DIR

done
