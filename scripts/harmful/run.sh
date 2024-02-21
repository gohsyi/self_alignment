#!/usr/bin/env bash

set -e

MODEL=${1:-"llama-7b"}
SAMPLES=${2:-64}
SAMPLES_GEN=${3:-0}
SRC=${4:-"discrimination,stereotype,injustice"}
EPOCHS=${5:-2}
PEER=${6:-0}

MODELS=(
    "facebook/opt-350m"
    "facebook/opt-1.3b"
    "facebook/opt-2.7b"
    "facebook/opt-6.7b"
)

EVAL_DOMAINS=(
    "discrimination,stereotype,injustice"
    "hate_speech,offensive_language"
    "non_violent_unethical_behavior"
    # "violence,aiding_and_abetting,incitement"
)

for MODEL in ${MODELS[@]}; do
    for DOMAIN in ${EVAL_DOMAINS[@]}; do
        DIR="output/$MODEL/self_alignment_v0_peer$PEER/$DOMAIN/${SAMPLES}_${SAMPLES_GEN}"

        mkdir -p $DIR/0/data

        # cp -r output/$MODEL/pretrain/$MODEL $DIR/0/sft
        cp data/$DOMAIN/$SAMPLES/train.json $DIR/0/data/train.json

        PRETRAIN_PATH=output/$MODEL/pretrain/model
        MODEL_PATH=$PRETRAIN_PATH

        # Learning rate is 2e-5
        LEARNING_RATE=0.00002

        for EPOCH in $(eval echo "{1..$EPOCHS}"); do
            echo "******* epoch $EPOCH *******"
            DATA_DIR=$DIR/$EPOCH/data
            SFT_DIR=$DIR/$EPOCH/sft
            EVAL_DIR=$DIR/$EPOCH/eval

            echo "Training with learning rate $LEARNING_RATE"
            echo "Writing to $SFT_DIR/output.log"
            mkdir -p $SFT_DIR

            if [ ! -f "$SFT_DIR/pytorch_model.bin" ]; then
                bash scripts/sft.sh \
                --model_name_or_path $MODEL_PATH \
                --datasets "local_json:1.0:$DATA_DIR/train.json" \
                --learning_rate $LEARNING_RATE \
                --offload all \
                --peer $PEER \
                --output_dir $SFT_DIR &> $SFT_DIR/output.log
            fi

            LEARNING_RATE=$(awk "BEGIN {printf \"%.10f\",${LEARNING_RATE}/2}")

            echo "Evaluating"

            if [ ! -f "$EVAL_DIR/$DOMAIN/eval.json" ]; then
                mkdir -p $EVAL_DIR/$DOMAIN
                cp data/$DOMAIN/eval.json $EVAL_DIR/$DOMAIN/eval.json

                echo "Writing to $EVAL_DIR/$DOMAIN/output.log"
                python3 self_alignment/generate.py \
                --model_name_or_path $SFT_DIR \
                --eval_path data/$DOMAIN/eval.json \
                --evaluate \
                --output_dir $EVAL_DIR/$DOMAIN &> $EVAL_DIR/$DOMAIN/output.log
            fi

            MODEL_PATH=$SFT_DIR
        done
    done
done
