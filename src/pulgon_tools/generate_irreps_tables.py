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
from typing import List, Tuple, Union

import numpy as np
from ase import Atom, Atoms
from ase.io import read
from pymatgen.core.operations import SymmOp

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer
from pulgon_tools.line_group_table import get_family_num_from_sym_symbol
from pulgon_tools.utils import (
    brute_force_generate_group_subsequent,
    find_axis_center_of_nanotube,
    get_character_num_withparities,
    get_character_withparities,
)


def get_linegroup_symmetry_dataset(
    poscar: Union[str, Atom, Atoms],
) -> Tuple[Atoms, int, int, float, List[SymmOp], List[List[int]]]:
    """Extract the full line group symmetry dataset from a structure.

    Detects the axial point group and cyclic group, then generates all
    symmetry operations via brute-force group multiplication.

    Args:
        poscar: path to a POSCAR file, or an ASE Atom/Atoms object.

    Returns:
        tuple of (atom_center, family, nrot, aL, ops_car_sym, order_ops).
    """
    if type(poscar) == str:
        atom = read(poscar)
    elif type(poscar) == Atom or type(poscar) == Atoms:
        atom = poscar
    else:
        print("Unknown input")

    atom_center = find_axis_center_of_nanotube(atom)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol

    trans_op = np.round(cyclic.get_generators(), 6)
    rots_op = np.round(obj.get_generators(), 6)

    if rots_op.size != 0:
        mats = np.vstack(([trans_op], rots_op))
    else:
        mats = trans_op.copy()
    ops, order_ops = brute_force_generate_group_subsequent(mats, symec=1e-2)

    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    family = get_family_num_from_sym_symbol(trans_sym, rota_sym)
    return atom_center, family, nrot, aL, ops_car_sym, order_ops


def main():
    """CLI entry point for computing irreps tables and character tables."""
    parser = argparse.ArgumentParser(
        description="Return the representation matrices or character table from a structure"
    )
    parser.add_argument(
        "-p", "--POSCAR", help="path to the file of a structure"
    )
    parser.add_argument(
        "-q",
        "--qpoint_z",
        type=float,
        default=0.0,
        help="The qpoint in the periodic direction (z), from 0 to 1",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for atomic positions",
    )
    parser.add_argument(
        "-s",
        "--savename_chara",
        default="characters",
        help="The filename of character",
    )
    parser.add_argument(
        "-r",
        "--enable_rep_matrix",
        action="store_true",
        help="Enable output of irreducible representation matrices",
    )

    args = parser.parse_args()
    st_name = args.POSCAR
    qpoint_z = args.qpoint_z
    symprec = args.tolerance
    chara_filename = args.savename_chara
    enable_rep_matrix = args.enable_rep_matrix

    atom = read(st_name)

    (
        atom_center,
        family,
        nrot,
        aL,
        ops_car_sym,
        order_ops,
    ) = get_linegroup_symmetry_dataset(atom)
    qp_normalized = qpoint_z / aL * 2 * np.pi
    DictParams = {
        "qpoints": qp_normalized,
        "nrot": nrot,
        "order": order_ops,
        "family": family,
        "a": aL,
    }

    if enable_rep_matrix:
        (
            characters,
            irreps_values,
            irreps_symbols,
        ) = get_character_num_withparities(DictParams, symprec=symprec)
        representation_mat, _, _ = get_character_withparities(
            DictParams, symprec=symprec
        )
        representation_mat_dict = {}
        for i, rep in enumerate(representation_mat):
            representation_mat_dict[f"D_irrep_{i}"] = rep

        np.savez(
            chara_filename,
            characters=characters,
            ireps_values=irreps_values,
            ireps_symbols=irreps_symbols,
            **representation_mat_dict,
        )

    else:

        (
            characters,
            irreps_values,
            irreps_symbols,
        ) = get_character_num_withparities(DictParams, symprec=symprec)

        np.savez(
            chara_filename,
            characters=characters,
            ireps_values=irreps_values,
            ireps_symbols=irreps_symbols,
        )


if __name__ == "__main__":
    main()
