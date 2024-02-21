import pandas as pd
import json
import random
import os
import tqdm
import argparse
from self_alignment.api import API
api = API()

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()

with open(args.path) as f:
    dataset = json.load(f)
for x in tqdm.tqdm(dataset):
    x['embedding'] = api.embedding(x['input'])
with open(args.path, 'w') as f:
    json.dump(dataset, f, indent=4)
    