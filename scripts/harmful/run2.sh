#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}
SAMPLES=${2:-64}
SAMPLES_GEN=${3:-512}
SRC=${4:-"mix"}
EPOCHS=${5:-3}
PEER=${6:-0}

EVAL_DOMAINS=(
    "discrimination,stereotype,injustice"
    "hate_speech,offensive_language"
    "non_violent_unethical_behavior"
    "violence,aiding_and_abetting,incitement"
)

ROOT="/mnt/bn/domain-adaptation/mlx/users/hongyi.guo/repo/5923/self-alignment"
DIR="$ROOT/output/$MODEL/self_alignment_v2_peer$PEER/$SRC/${SAMPLES}_${SAMPLES_GEN}"

mkdir -p $DIR/0/data
mkdir -p $DIR/1/data
cp $ROOT/data/$SRC/$SAMPLES/train.json $DIR/0/data/train.json
cp -r $ROOT/output/$MODEL/pretrain/$MODEL $DIR/0/sft

python3 self_alignment/generate.py \
--model_name_or_path $ROOT/output/$MODEL/pretrain/model \
--generate $SAMPLES_GEN \
--dataset_paths $DIR/0/data/train.json \
--do_sample \
--batch_size 4 \
--verbose \
--threshold_rouge 0.7 \
--threshold_length 5 \
--filter \
--output_dir $DIR/1/data &> $DIR/1/data/output.log

for EPOCH in $(eval echo "{2..$EPOCHS}"); do
    mkdir -p $DIR/$EPOCH/data
    cp -r $DIR/1/data/train.json $DIR/$EPOCH/data/train.json
done

# Learning rate is 2e-5
LEARNING_RATE=0.00002

for EPOCH in $(eval echo "{1..$EPOCHS}"); do
    echo "******* epoch $EPOCH *******"
    DATA_DIR=$DIR/$EPOCH/data
    SFT_DIR=$DIR/$EPOCH/sft
    EVAL_DIR=$DIR/$EPOCH/eval

    MODEL_PATH=$DIR/$(( EPOCH - 1 ))/sft

    echo "Generating dataset"
    echo "Writing to $DATA_DIR/output.log"
    mkdir -p $DATA_DIR
    
    python3 self_alignment/generate.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_paths $(eval echo "$DIR/{0..$(( EPOCH - 1))}/data/train.json") \
    --do_sample \
    --eval_path $DATA_DIR/train.json \
    --batch_size 4 \
    --threshold_rouge 0.7 \
    --threshold_length 5 \
    --filter \
    --output_dir $DATA_DIR &> $DATA_DIR/output.log

    echo "Training with learning rate $LEARNING_RATE"
    echo "Writing to $SFT_DIR/output.log"
    mkdir -p $SFT_DIR

    bash scripts/sft.sh \
    --model_name_or_path $MODEL_PATH \
    --datasets "local_json:1.0:$DIR/0/data/train.json&$DATA_DIR/eval.json" \
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
done
