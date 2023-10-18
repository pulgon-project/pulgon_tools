# Copyright 2023 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http:#www.apache.org/licenses/LICENSE-2.0
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
from ase.io.vasp import read_vasp, write_vasp

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)


def get_symcell(monomer):

    set_trace()


def main():
    poscar = read_vasp("12-12-AM.vasp")
    cyclic = CyclicGroupAnalyzer(poscar, corner=True, symprec=0.01)
    cy, mon = cyclic.get_cyclic_group()

    get_symcell(mon[0])


if __name__ == "__main__":
    main()
