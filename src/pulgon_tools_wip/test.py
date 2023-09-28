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
