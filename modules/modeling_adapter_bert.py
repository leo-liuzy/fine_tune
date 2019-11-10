# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

from ipdb import set_trace as bp
import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings
from .modeling_bert import *


class BertFeedForwardAdapter(nn.Module):
    def __init__(self, input_size, hidden_size=64, init_scale=1e-3):
        super(AdapterBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_scale = init_scale
        self.compress = nn.Linear(input_size, hidden_size)
        self.decompress = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.compress(x)
        x = gelu(x)
        x = self.decompress(x)
        return x
