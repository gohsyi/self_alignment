set -e


for MODEL in "llama-7b" "llama2-7b" "opt-6.7b"; do
    echo "$MODEL"

    PRETRAIN_DIR=output/$MODEL/pretrain/alpaca
    python3 scripts/alpaca/converter.py $PRETRAIN_DIR/eval.json

    OUTPUT_DIRS=(
        "output/$MODEL/icl/64/alpaca"
        "output/$MODEL/icl_random/64/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_0/1/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_0/2/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_0/3/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_0/4/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_512/1/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_512/2/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_512/3/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_512/4/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_1024/1/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca/64_1024/2/eval/alpaca"
        # "output/$MODEL/self_alignment_v0_peer0/alpaca/64_1024/3/eval/alpaca"
        # "output/$MODEL/self_alignment_v0_peer0/alpaca/64_1024/4/eval/alpaca"
        "output/$MODEL/icl/64/alpaca_6"
        "output/$MODEL/icl_random/64/alpaca_6"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_0/1/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_0/2/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_0/3/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_0/4/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_512/1/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_512/2/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_512/3/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_512/4/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_1024/1/eval/alpaca"
        "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_1024/2/eval/alpaca"
        # "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_1024/3/eval/alpaca"
        # "output/$MODEL/self_alignment_v0_peer0/alpaca_6/64_1024/4/eval/alpaca"
    )
    
    for OUTPUT_DIR in ${OUTPUT_DIRS[@]}; do
        echo $OUTPUT_DIR
        if [ -f $OUTPUT_DIR/eval.json ] && [ ! -f $OUTPUT_DIR/leaderboard.csv ]; then
            python3 scripts/alpaca/converter.py $OUTPUT_DIR/eval.json
            alpaca_eval --model_outputs $OUTPUT_DIR/eval.json --reference_outputs $PRETRAIN_DIR/eval.json
        fi
    done
done
