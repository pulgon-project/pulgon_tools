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

import numpy as np
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from numpy.linalg.linalg import eigvals
from sympy.printing.octave import print_octave_code

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer, get_symcell
from pulgon_tools.line_group_table import get_family_Num_from_sym_symbol
from pulgon_tools.utils import get_symbols_from_ops


def test_symcell(poscar):
    obj = CyclicGroupAnalyzer(poscar)
    res = get_symcell(obj.monomers[0])
    write_vasp("poscar.vasp", res)
    return res


if __name__ == "__main__":
    # poscar = read_vasp("../../test/data/9-9-AM")
    # poscar = read_vasp("../../test/data/12-12-AM")
    # poscar = read_vasp("../../test/data/24-0-ZZ")
    # poscar = read_vasp("../../test/data/C4h")
    # poscar = read_vasp("../../test/data/C4v")
    poscar = read_vasp("POSCAR-defect-pri-5.vasp")
    # poscar = read_vasp("../../test/data/C4")
    test_symcell(poscar)
