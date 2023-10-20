# Copyright 2023 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import json
import pickle
import typing
from ast import literal_eval
from pdb import set_trace

import numpy as np

# class CharacterDataset(typing.NamedTuple):
#     """dataset of the final result
#
#     Args:
#         index: order
#         quantum_number: SAB
#         character_table:
#     """
#
#     index: list[int]
#     quantum_number: list[tuple]
#     character_table: list


# with open('test.pkl', 'rb') as handle:
#     b = pickle.load(handle)
res = np.load("test.npy", allow_pickle=True)

set_trace()
