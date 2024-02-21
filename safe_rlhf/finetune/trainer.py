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
"""Trainer class for supervised finetuning."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from safe_rlhf.datasets import SupervisedDataset
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import get_all_reduce_mean



class SupervisedFinetuneTrainer(SupervisedTrainer):
    """Trainer class for supervised finetuning."""

    TRAINING_TYPE = 'sft'
    DATASET_TYPE = SupervisedDataset
    MODEL_TYPE = AutoModelForCausalLM

    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            'loss': outputs.loss
        }

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        """Performs a single training step.

        Args:
            input_ids (torch.LongTensor): input ids for causal inputs to complete with.
            labels (torch.LongTensor): labels for the full sequence.
            attention_mask (torch.BoolTensor): attention mask for the labels.

        Returns:
            dict[str, Any]: training loss, learning rate
        """
        loss = self.loss(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }


class SupervisedFinetuneTrainerWithFilter(SupervisedFinetuneTrainer):
    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        """Filter Loss"""
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).reshape(shift_labels.size()).mean(-1)

        filtered_loss = loss[loss >= self.args.loss_threshold]
        print("After filtering, {}/{} samples remain".format(len(filtered_loss), len(loss)))
        return {
            'loss': filtered_loss.mean(),
        }


class SupervisedFinetuneTrainerWithPeerLoss(SupervisedFinetuneTrainer):
    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        """Peer Loss"""
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B, L, N = shift_logits.shape
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).reshape((B, L)).mean(-1)

        peer_term = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels[torch.randperm(B)].view(-1),
        ).reshape((B, L)).mean(-1)

        loss -= self.args.peer * peer_term

        return {
            'loss': loss.mean(),
        }
