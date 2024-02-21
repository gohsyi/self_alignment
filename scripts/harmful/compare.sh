set -e

python self_alignment/compare.py \
output/llama-7b/self_alignment_v0_peer0/discrimination,stereotype,injustice/64_512/2/eval/discrimination,stereotype,injustice/eval.json \
--cmp_with \
output/llama-7b/self_alignment_v0_peer0/discrimination,stereotype,injustice/64_1024/1/eval/discrimination,stereotype,injustice/eval.json \
output/llama-7b/pretrain/discrimination,stereotype,injustice/eval.json \
output/llama-7b/icl/64/discrimination,stereotype,injustice/eval.json
