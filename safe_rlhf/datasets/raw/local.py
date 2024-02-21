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
"""Stanford Alpaca dataset for supervised instruction fine-tuning."""

from __future__ import annotations

import csv
import json
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ['LocalJsonDataset']


class LocalJsonDataset(RawDataset):
    NAME: str = 'local_json'

    def __init__(self, path: str | None = None) -> None:
        self.data = []
        for p in path.split('&'):
            with open(p) as f:
                self.data.extend(json.load(f))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(input=data['input'], answer=data['answer'])

    def __len__(self) -> int:
        return len(self.data)


class LocalCSVDataset(RawDataset):
    NAME: str = 'local_csv'

    def __init__(self, path: str | None = None) -> None:
        with open(path) as f:
            self.data = csv.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data.iloc[index]
        return RawSample(input=data['input'], answer=data['answer'])

    def __len__(self) -> int:
        return len(self.data)
