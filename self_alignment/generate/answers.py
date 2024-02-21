from __future__ import annotations

import argparse
import os
import json
import torch
import pandas as pd

from tqdm import tqdm
from random import sample
from sklearn.neighbors import NearestNeighbors

from safe_rlhf.evaluate.bigbench.model import SafeRLHFModel as InferenceModel

from self_alignment.config import Config
from self_alignment.api import API
from self_alignment.utils import rouge, load_json, format_context, postprocessing
from self_alignment.evaluation import evaluate, summarize

# PROMPT_REGEX = re.compile(r'(.*?)(?=Assistant:|$|<\|endoftext\|>)', re.DOTALL)
# RESPONSE_REGEX = re.compile(r'(.*?)(?=Human:|$|<\|endoftext\|>)', re.DOTALL)

device = torch.device('cuda:0')
api = API()


def parse_args():
    parser = argparse.ArgumentParser(description='Collect responses for eval set')
    parser.add_argument('--model_name_or_path', type=str, help='Path to model')
    parser.add_argument('--eval_path', type=str, help='Path to test data')
    parser.add_argument('--dataset_paths', nargs='*', default=[], help='Paths to datasets')
    parser.add_argument('--context_nums', nargs='*', type=int, default=[], help='',)
    parser.add_argument('--batch_size', type=int, default=4, help='',)
    parser.add_argument('--random', action='store_true', help='Random instead of KNN')
    parser.add_argument('--verbose', action='store_true', help='Auxiliary outputs')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=10)
    parser.add_argument('--threshold_length', type=int, default=0)
    parser.add_argument('--filter', action='store_true', help='Whether to filter the generated dataset')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to output')
    parser.add_argument('--save_name', type=str, default='eval.json', help='Saving name')
    args = parser.parse_args()
    return args


def generate(
    cfg: dict,
    model: InferenceModel,
    eval_dataset: list[dict], 
    input_datasets: list[dict], 
    context_nums: int, 
    knn: bool = True,
) -> list[dict]:
    
    prompts = []  # Prompts for generating inputs
    for data in eval_dataset:
        contexts = []  # Contexts for ICL
        y = [data['embedding']]  # Embedding of the input
        for dataset, num in zip(input_datasets, context_nums):
            if knn:
                X = [x['embedding'] for x in dataset]
                _, indices = NearestNeighbors(n_neighbors=num).fit(X).kneighbors(y)
                seeds = [dataset[i] for i in indices[0]]
            else:
                seeds = sample(dataset, num)
            for x in seeds:
                contexts.extend((x['input'], x['answer']))
        contexts.append(data['input'])
        prompts.append(format_context(contexts))

    # Generate
    responses = model.generate_text(prompts, prompt_input='{input}', **cfg)

    # Postprocessing
    for data, response in zip(eval_dataset, responses):
        data['answer'] = postprocessing(response)
    return eval_dataset


def main() -> None:
    args = parse_args()
    print(args)

    if not args.context_nums and args.dataset_paths:
        context_nums = [1 for _ in args.dataset_paths]
        context_nums[0] += 8 - sum(context_nums)
    else:
        context_nums = args.context_nums

    input_datasets = []
    for path in args.dataset_paths:
        input_datasets.append(load_json(path))

    # Wrap the model for batched inference
    model = InferenceModel(
        args.model_name_or_path,
        show_progress=True,
        batch_size=args.batch_size,
    )
    cfg = Config(args, model)

    # Generating evaluation dataset
    eval_dataset = load_json(args.eval_path)
    eval_dataset = generate(
        cfg.inference, model, eval_dataset, input_datasets, context_nums, knn=not args.random
    )
    pd.DataFrame.from_records(eval_dataset).to_csv(os.path.join(args.output_dir, 'generated_answers.csv'))

    output_dataset = []
    for data in eval_dataset:
        if not args.filter or (
            len(data['answer'].split()) >= args.threshold_length
            and data['input'] not in data['answer']
        ):
            output_dataset.append(data)

    if args.evaluate:
        rewards, costs, moderation = evaluate(output_dataset, batch_size=args.batch_size)
        for data, dam, reward, cost in zip(output_dataset, moderation, rewards, costs):
            data['harmful'] = dam['categories']
            data['cost'] = cost.item()
            data['reward'] = reward.item()
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(summarize(output_dataset), f, indent=4)
        
    with open(os.path.join(args.output_dir, args.save_name), 'w') as f:
        json.dump(output_dataset, f, indent=4)


if __name__ == '__main__':
    main()
