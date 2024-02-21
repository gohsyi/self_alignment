import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)

args = parser.parse_args()

with open(args.file, 'r') as f:
    data = json.load(f)
for x in data:
    if 'input' in x:
        x['instruction'] = x.pop('input')
    if 'answer' in x:
        x['output'] = x.pop('answer')
    if 'embedding' in x:
        del x['embedding']
with open(args.file, 'w') as f:
    json.dump(data, f, indent=4)
