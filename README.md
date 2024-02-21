# Retrieval-Augmented Self-Alignment (RASA)

## Instrallation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## RASA Quickstart

Training scripts are located in `scripts/`. Three datasets: `PKU-SafeRLHF`,`TruthfulQA`, and `Alpaca-Eval`, corresponding to `scripts/harmful`, `scripts/truthful`, and `scripts/alpaca` respectively. For example, if you want to run the training script for the `PKU-SafeRLHF` dataset, you would run:

```bash
bash scripts/harmful/run.sh <MODEL> <SAMPLES> <SAMPLES_GEN> <SRC> <EPOCHS>
```

Explanation:

`<MODEL>`: the model you want to train (`llama-7b`, `llama2-7b`, `opt-6.7b`)

`<SAMPLES>`: the number of labeled samples

`<SAMPLES_GEN>`: the number of generated samples. Setting to 0 for SFT.

`<SRC>`: domain

`<EPOCHS>`: the number of training epochs. Note that you have to prepare the data first (instructions coming soon).

The `run.sh` script runs the training process, and automatically does the evaluation for you in three domains: `discrimination,stereotype,injustice`, `hate_speech,offensive_language`, and `non_violent_unethical_behavior`. It's similar for truthful_qa dataset. But for alpaca dataset, you have to manually run the evaluation script `scripts/alpaca/eval.sh` and specify the output dirs in the script (because the evaluation program does not support the azure api used in merlin lq cluster)


## Evaluation

Evaluation scripts of the pretrain/icl_knn/icl_random are all located in the `scripts/`.
