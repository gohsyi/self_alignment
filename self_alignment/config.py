from __future__ import annotations

from argparse import Namespace
from transformers import GenerationConfig

from safe_rlhf.evaluate.bigbench.model import SafeRLHFModel as InferenceModel


class Config:
    def __init__(self, args: Namespace, model: InferenceModel) -> None:
        self.args = args
        self._bos_token_id = model.bos_token_id
        self._eos_token_id = model.eos_token_id
        self._pad_token_id = model.pad_token_id
        self._encoding_config = {
            'return_tensors': 'pt',
            'padding': True,
            'truncation': True,
        }
        self._decoding_config = {
            'clean_up_tokenization_spaces': True,
            'skip_special_tokens': True
        }
    
    @property
    def inference(self) -> dict:
        """Testing time""" 
        return {
            'encoding_config': self._encoding_config,
            'decoding_config': self._decoding_config,
            'generation_config': GenerationConfig(max_new_tokens=200),
        }

    @property
    def answer(self) -> dict:
        return {
            'encoding_config': self._encoding_config,
            'decoding_config': self._decoding_config,
            'generation_config': GenerationConfig(
                bos_token_id=self._bos_token_id,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
                max_new_tokens=100,
                temperature=1,
                do_sample=True,
                num_beams=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=10,
                exponential_decay_length_penalty=(30, 1.05),
            )
        }

    @property
    def input(self) -> dict:
        return {
            'encoding_config': self._encoding_config,
            'decoding_config': self._decoding_config,
            'generation_config': GenerationConfig(
                bos_token_id=self._bos_token_id,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
                max_new_tokens=200,
                do_sample=True,
                num_beams=5,
                repetition_penalty=1.05,
                no_repeat_ngram_size=10,
                length_penalty=2,
                exponential_decay_length_penalty=(15, 1.6),
            ),
        }
