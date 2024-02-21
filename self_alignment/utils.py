from __future__ import annotations

import os
import json
import csv
import pandas as pd
from safe_rlhf.configs import (
    PROMPT_ASSISTANT, 
    PROMPT_BEGIN, 
    PROMPT_END, 
    PROMPT_USER, 
    PROMPT_INPUT
)
import evaluate
rouge = evaluate.load('rouge')


def load_json(file_path: str | os.PathLike) -> dict:
    root, ext = os.path.splitext(file_path)
    if ext == '.json':
        with open(file_path) as f:
            dataset = json.load(f)
    elif ext == '.csv':
        dataset = json.loads(pd.read_csv(file_path).to_json(orient='records'))
    else:
        raise ValueError
    return dataset


def format_context(input: list[str]) -> str:  # pylint: disable=redefined-builtin
    buffer = []
    for i, line in enumerate(input):
        if i % 2 == 0:
            # User input
            buffer.append(PROMPT_INPUT.format(input=line))
        else:
            # Assistant response
            buffer.extend((' ', line, ' ', PROMPT_END, '\n\n'))
    if len(input) % 2 == 0:
        buffer.extend((PROMPT_BEGIN, PROMPT_USER.format(input='').rstrip()))
    return ''.join(buffer)


def postprocessing(
    response: str, 
    keywords: list[str] = [
        'BEGIN', 'END', 
        'USER', 'ASSISTANT', 'ASSISTENT', 
        '\\begin', '\\end', 
        'ANS'
    ]
) -> str:
    for keyword in keywords:
        index = response.find(keyword)
        if index > 0:
            response = response[: index]
    return response.strip()
