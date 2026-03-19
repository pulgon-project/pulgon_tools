# Copyright 2023-2026 The PULGON Project Developers
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

from ase.io.vasp import read_vasp

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer
from pulgon_tools.line_group_table import get_family_num_from_sym_symbol
from pulgon_tools.utils import find_axis_center_of_nanotube

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect the line group of a nanostructure"
    )

    parser.add_argument("data_poscar", help="poscar")
    parser.add_argument("tol", help="tolerance", type=float, default=1e-3)

    args = parser.parse_args()

    path_poscar = args.data_poscar
    tol = float(args.tol)

    poscar_ase = read_vasp(path_poscar)
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)

    atom = cyclic._primitive
    atom_center = find_axis_center_of_nanotube(atom)

    obj = LineGroupAnalyzer(atom_center, tolerance=tol)
    nrot = obj.get_rotational_symmetry_number()

    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol
    family = get_family_num_from_sym_symbol(trans_sym, rota_sym)

    print("family=", family)
    print("generalized translation:", cyclic.cyclic_group)
    print("axial point group:", obj.sch_symbol)
