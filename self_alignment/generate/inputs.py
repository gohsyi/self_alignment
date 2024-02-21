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
    parser.add_argument('--generate', type=int, default=0, help='Number of questions to generate')
    parser.add_argument('--dataset_paths', nargs='*', default=[], help='Paths to datasets',)
    parser.add_argument('--batch_size', type=int, default=4, help='Inference batch size')
    parser.add_argument('--context_nums', nargs='*', type=int, default=[], help='Example number of each dataset')
    parser.add_argument('--random', action='store_true', help='Random instead of KNN')
    parser.add_argument('--do_sample', action='store_true', help='Generation configuration')
    parser.add_argument('--max_length', type=int, default=2048, help='Generation configuration')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Generation configuration')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature')
    parser.add_argument('--num_beams', type=int, default=1, help='Generation configuration')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Generation configuration')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=10, help='Generation configuration')
    parser.add_argument('--threshold_length', type=int, default=0, help='Keep if length >= threshold_length')
    parser.add_argument('--threshold_rouge', type=float, default=1.0, help='Keep if rouge >= threshold_rouge')
    parser.add_argument('--filter', action='store_true', help='Whether to filter the generated dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to output')
    parser.add_argument('--save_name', type=str, default='train.json', help='Name of the saved file')
    args = parser.parse_args()
    return args


def generate(
    cfg: dict,
    model: InferenceModel,
    input_datasets: list[dict], 
    context_nums: list[int], 
    num_generations: int,
) -> list[dict]:
    
    prompts = []  # Prompts for generating inputs
    output_dataset = []
    inputs = []  # All existing inputs
    for _ in range(num_generations):
        contexts = []  # Contexts sampled for current input
        categories = []  # Categories of those contexts
        # Sample num samples from dataset
        for dataset, num in zip(input_datasets, context_nums):
            inputs.extend(data['input'] for data in dataset)
            selection = sample(dataset, num)
            for seed in selection:
                contexts.extend((seed['input'], seed['answer']))
                categories.append(seed['category'])

        # Log contexts
        data = {}
        for i, ctx in enumerate(contexts[::2]):
            data['context_{}'.format(i + 1)] = ctx
        # Majority voting for the category
        data['category'] = max(categories, key=categories.count)
        output_dataset.append(data)
        # Add 'BEGINNING OF CONVERSATION: USER:' to the end of the contexts 
        # to prompt LLM to generate next input
        prompts.append(format_context(contexts))

    # Generate
    responses = model.generate_text(prompts, prompt_input='{input}', **cfg)

    # Postprocessing
    for data, response in tqdm(zip(output_dataset, responses), desc='Postprocessing'):
        # Cut off when LLM generates answer for the generated input
        data['input'] = postprocessing(response)
        data['embedding'] = api.embedding(data['input'])
        # ROUGE-L score for filtering out novel inputs
        data.update(rouge.compute(
            predictions=[data['input']], 
            references=[[value for key, value in data.items() if key.startswith('context_')]],
            rouge_types=['rougeL'],
        ))

    return output_dataset


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

    # Generating training dataset
    dataset = generate(cfg.input, model, input_datasets, context_nums, args.generate)
    pd.DataFrame.from_records(dataset).to_csv(os.path.join(args.output_dir, 'generated_inputs.csv'))

    # Filter question dataset
    output_dataset = []
    for data in dataset:
        if not args.filter or (
            all(data['input'] != x['input'] for x in output_dataset) 
            and data['rougeL'] <= args.threshold_rouge 
            and len(data['input'].split()) >= args.threshold_length
        ):
            output_dataset.append(data)

    with open(os.path.join(args.output_dir, args.save_name), 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == '__main__':
    main()
