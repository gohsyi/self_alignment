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
    parser.add_argument('--generate', type=int, default=0, help='')
    parser.add_argument('--dataset_paths', nargs='*', default=[], help='Paths to datasets',)
    parser.add_argument('--batch_size', type=int, default=2, help='',)
    parser.add_argument('--context_nums', type=int, default=8, help='',)
    parser.add_argument('--random', action='store_true', help='Random instead of KNN')
    parser.add_argument('--verbose', action='store_true', help='Auxiliary outputs')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to output')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=10)
    parser.add_argument('--threshold_rouge', type=float, default=1.0)
    parser.add_argument('--threshold_length', type=int, default=0)
    parser.add_argument('--filter', action='store_true', 
                        help='Answer should not repeat the input. Do not set this for evaluation')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    return args


def generate_inputs(
    cfg: dict,
    model: InferenceModel,
    input_datasets: list[dict], 
    context_nums: list[int], 
    num_generations: int,
    rouge_threhold: float = 0.7,
) -> list[dict]:
    
    prompts = []  # Prompts for generating inputs
    output_dataset = []
    inputs = []  # All existing inputs
    for _ in range(num_generations):
        contexts = []  # Contexts sampled for current input
        # categories = []  # Categories of those contexts
        # Sample num samples from dataset
        for dataset, num in zip(input_datasets, context_nums):
            inputs.extend(data['input'] for data in dataset)
            selection = sample(dataset, num)
            for seed in selection:
                if model.num_tokens(format_context(contexts + [seed['input'], seed['answer']])) < 2000:
                    contexts.extend((seed['input'], seed['answer']))
                # categories.append(seed['category'])

        # Log contexts
        data = {}
        for i, ctx in enumerate(contexts[::2]):
            data['context_{}'.format(i + 1)] = ctx
        # Majority voting for the category
        # data['category'] = max(categories, key=categories.count)
        output_dataset.append(data)
        # Add 'BEGINNING OF CONVERSATION: USER:' to the end of the contexts 
        # to prompt LLM to generate next input
        prompts.append(format_context(contexts))

    # Generate
    responses = model.generate_text(prompts, prompt_input='{input}', **cfg)

    # Postprocessing
    inputs_ = []  # Generated inputs so far
    for data, response in tqdm(zip(output_dataset, responses), desc='Postprocessing'):
        # Cut off when LLM generates answer for the generated input
        data['input'] = postprocessing(response)
        # ROUGE-L score for filtering out novel inputs
        data['rougeL'] = rouge.compute(
            predictions=[data['input']], 
            references=[[value for key, value in data.items() if key.startswith('context_')]],
            # references=[inputs + inputs_],
            rouge_types=['rougeL']
        )['rougeL']
        data['embedding'] = api.embedding(data['input'])
        if data['rougeL'] <= rouge_threhold:
            inputs.append(data['input'])

    return output_dataset


def generate_answers(
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

    context_nums = [1 for _ in args.dataset_paths]
    if context_nums:
        context_nums[0] += args.context_nums - sum(context_nums)

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

    if args.eval_path:
        # Generating evaluation dataset
        eval_dataset = load_json(args.eval_path)
        eval_dataset = generate_answers(
            cfg.inference, model, eval_dataset, input_datasets, context_nums, knn=not args.random
        )
        save_name = 'eval.json'
    else:
        # Generating training dataset
        eval_dataset = generate_inputs(
            cfg.input, model, input_datasets, context_nums, args.generate, args.threshold_rouge
        )
        pd.DataFrame.from_records(eval_dataset).to_csv(
            os.path.join(args.output_dir, 'generated.csv')
        )
        eval_dataset = generate_answers(
            cfg.answer, model, eval_dataset, input_datasets, context_nums, knn=not args.random
        )
        save_name = 'train.json'

    output_dataset = []
    for data in eval_dataset:
        if not args.filter or (
            all(data['input'] != x['input'] for x in output_dataset) 
            and data['rougeL'] <= args.threshold_rouge 
            and len(data['input'].split()) >= args.threshold_length
            and len(data['answer'].split()) >= args.threshold_length
            and data['answer'] not in data['input']
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
        
    with open(os.path.join(args.output_dir, save_name), 'w') as f:
        json.dump(output_dataset, f, indent=4)


if __name__ == '__main__':
    main()
