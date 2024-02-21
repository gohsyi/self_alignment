# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import scipy
from tqdm import tqdm

from transformers import GenerationConfig

# isort: off
import torch  # should be imported before bigbench
import bigbench.api.model as model
import bigbench.models.model_utils as model_utils

# isort: on
from safe_rlhf.configs import PROMPT_INPUT
from safe_rlhf.models.pretrained import load_pretrained_models


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_masks: torch.Tensor,
) -> list[float]:
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_label_masks = label_masks[..., 1:].contiguous()

    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).reshape(shift_labels.size())

    loss = (loss * shift_label_masks).sum(-1)
    return (-loss).cpu().numpy().tolist()


class SafeRLHFModel(model.Model):
    def __init__(
        self,
        model_path: str,
        show_progress: bool = False,
        batch_size: int = 4
    ) -> None:
        self._model, self._tokenizer = load_pretrained_models(
            model_path,
            padding_side='left',
            auto_device_mapping=True,
            trust_remote_code=True,
        )
        if 'llama2' in model_path:
            self._model = self._model.bfloat16()
        self._show_progress = show_progress
        self._batch_size = batch_size
    
    def num_tokens(self, text: str) -> int:
        return len(self._tokenizer(text).input_ids)

    @property
    def bos_token(self):
        return self._tokenizer.bos_token
    
    @property
    def bos_token_id(self):
        return self._tokenizer.bos_token_id

    @property
    def eos_token(self):
        return self._tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id
    
    @property
    def pad_token(self):
        return self._tokenizer.pad_token
    
    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    def model_data(self):
        return model.ModelData(
            model_family='Llama',
            model_name='alpaca7B',
            non_embedding_params=0,
            flop_matched_non_embedding_params=0,
            total_params=7_000_000_000,
            training_batch_size=128,
            training_steps=500_000 / 128 * 5,
            description='Alpaca 13B',
            decoding_params={},
        )

    def generate_text(
        self,
        inputs: str | list[str],
        max_length: int | None = 2048,
        stop_string: str | None = None,
        output_regex: str | None = None,
        prompt_input: str = PROMPT_INPUT,
        generation_config: GenerationConfig | None = None,
        encoding_config: dict | None = None,
        decoding_config: dict | None = None,
        device: str | torch.device = torch.device('cuda:0')
    ) -> str | list[str]:
        
        max_length = max_length or self._max_length
        original_inputs = inputs
        inputs = inputs if isinstance(inputs, list) else [inputs]
        inputs = [prompt_input.format(input=text) for text in inputs]
        generated = []

        idx_lst = range(0, len(inputs), self._batch_size)
        if self._show_progress:
            idx_lst = tqdm(idx_lst, desc='Generating text...')

        for idx in idx_lst:
            batch_text = inputs[idx: idx + self._batch_size]
            batch = self._tokenizer(batch_text, **encoding_config).to(device)
            output = self._model.generate(**batch, generation_config=generation_config)
            # For debug only
            print('Raw:')
            for raw in self._tokenizer.batch_decode(output):
                print(raw)
            output_text = self._tokenizer.batch_decode(output, **decoding_config)
            for i, text in enumerate(output_text):
                # Filter out special tokens and spaces
                batch_text = self._tokenizer.batch_decode(batch.input_ids, **decoding_config)
                output_text[i] = text[len(batch_text[i]) :]
            generated.extend(output_text)

        if isinstance(original_inputs, str):
            generated = generated[0]

        return model_utils.postprocess_output(
            generated,
            max_length,
            stop_string,
            output_regex,
        )

    def cond_log_prob(
        self,
        inputs: str | list[str],
        targets: list[str] | list[list[str]],
        batch_size: int | None = None,
        absolute_normalization: bool | None = None,
    ) -> list[float] | list[list[float]]:
        batch_size = batch_size or self._batch_size
        original_inputs = inputs

        if isinstance(inputs, str):
            inputs = [inputs]
            targets = [targets]

        inputs = [PROMPT_INPUT.format(text) for text in inputs]

        flatten_idx, flatten_inputs, flatten_choices = [], [], []
        # pylint: disable-next=redefined-builtin
        for idx, (input, choice) in enumerate(zip(inputs, targets)):
            for choice_idx, choice_text in enumerate(choice):
                flatten_idx.append((idx, choice_idx))
                flatten_inputs.append(input)
                flatten_choices.append(choice_text)
        loss = []

        idx_lst = range(0, len(flatten_inputs), batch_size)
        if self._show_progress:
            idx_lst = tqdm(idx_lst, desc='Computing log prob...')

        for idx in idx_lst:
            batch_inputs = flatten_inputs[idx : idx + batch_size]
            batch_choices = flatten_choices[idx : idx + batch_size]
            batch = {
                'input_ids': [],
                'token_type_ids': [],
            }
            for input, choice in zip(batch_inputs, batch_choices):
                input_ids = self._tokenizer(
                    input,
                    truncation=True,
                    max_length=self._max_length,
                )['input_ids']
                choice_ids = self._tokenizer(
                    choice,
                    truncation=True,
                    max_length=self._max_length,
                )['input_ids']
                batch['input_ids'].append(input_ids + choice_ids)
                batch['token_type_ids'].append([0] * len(input_ids) + [1] * len(choice_ids))

            batch = self._tokenizer.pad(batch)
            batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

            with torch.no_grad():
                batch_logits = self._model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                ).logits

            batch_loss = _compute_loss(
                logits=batch_logits,
                labels=batch['input_ids'],
                label_masks=batch['token_type_ids'],
            )
            loss.extend(batch_loss)

        scores = [[] for _ in range(len(inputs))]
        for (idx, _), score in zip(flatten_idx, loss):
            assert score != 0, 'score should not be zero'
            scores[idx].append(score)

        if not absolute_normalization:
            scores = [list(score_row - scipy.special.logsumexp(score_row)) for score_row in scores]

        if isinstance(original_inputs, str):
            scores = scores[0]

        return scores
