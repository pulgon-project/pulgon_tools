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
from typing import Dict, List, Tuple, Union

import numpy as np
from ase import Atom, Atoms
from ase.io import read
from pymatgen.core.operations import SymmOp

from pulgon_tools.cli import RawDescriptionDefaultsHelpFormatter
from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer
from pulgon_tools.line_group_table import get_family_num_from_sym_symbol
from pulgon_tools.symmetry_projector import (
    DEFAULT_LAYER_TOLERANCE,
    DEFAULT_MATRIX_TOLERANCE,
    DEFAULT_SYMMETRY_TOLERANCE,
    GENERATOR_ROUND_DECIMALS,
    _extract_generator_angles,
    _extract_screw_parameters,
    _normalize_generators_for_irrep_table,
)
from pulgon_tools.utils import (
    brute_force_generate_group_subsequent,
    find_axis_center_of_nanotube,
    get_character_num_withparities,
    get_character_withparities,
)

LineGroupDataset = Tuple[
    Atoms,
    int,
    int,
    float,
    List[SymmOp],
    List[List[int]],
    Dict[str, Union[float, int]],
]


def get_linegroup_symmetry_dataset(
    poscar: Union[str, Atom, Atoms],
    tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    layer_tolerance: float = DEFAULT_LAYER_TOLERANCE,
    matrix_tolerance: float = DEFAULT_MATRIX_TOLERANCE,
) -> LineGroupDataset:
    """Extract the full line group symmetry dataset from a structure.

    Detects the axial point group and cyclic group, then generates all
    symmetry operations via brute-force group multiplication.

    Args:
        poscar: path to a POSCAR file, or an ASE Atom/Atoms object.
        tolerance: tolerance used for line-group symmetry detection.
        layer_tolerance: fractional z-layer tolerance for cyclic-group
            monomer translation candidates.
        matrix_tolerance: tolerance for matrix classification and group
            closure during operation generation.

    Returns:
        tuple of (atom_center, family, nrot, aL, ops_car_sym,
        order_ops, gen_angles).
    """
    if isinstance(poscar, str):
        atom = read(poscar)
    elif isinstance(poscar, (Atom, Atoms)):
        atom = poscar
    else:
        raise TypeError("poscar must be a path string, ASE Atom, or ASE Atoms")

    atom_center = find_axis_center_of_nanotube(atom)
    obj = LineGroupAnalyzer(
        atom_center,
        tolerance=tolerance,
        matrix_tolerance=matrix_tolerance,
    )
    cyclic = CyclicGroupAnalyzer(
        atom_center,
        tolerance=tolerance,
        layer_tolerance=layer_tolerance,
    )
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    cyclic_groups, _ = cyclic.get_cyclic_group()
    trans_sym = cyclic_groups[0]
    rota_sym = obj.sch_symbol
    family = get_family_num_from_sym_symbol(trans_sym, rota_sym)

    trans_op = np.round(cyclic.get_generators(), GENERATOR_ROUND_DECIMALS)
    rots_op = np.round(obj.get_generators(), GENERATOR_ROUND_DECIMALS)
    mats = _normalize_generators_for_irrep_table(
        family, nrot, trans_op, rots_op
    )
    ops, order_ops = brute_force_generate_group_subsequent(
        mats, symprec=matrix_tolerance
    )

    gen_angles: Dict[str, Union[float, int]] = _extract_generator_angles(
        mats, matrix_tolerance=matrix_tolerance
    )
    gen_angles.update(_extract_screw_parameters(trans_sym))

    ops_car_sym: List[SymmOp] = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    return atom_center, family, nrot, aL, ops_car_sym, order_ops, gen_angles


def main() -> None:
    """CLI entry point for computing irreps tables and character tables."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate line-group character tables, and optionally "
            "representation matrices, from a 1D periodic structure."
        ),
        epilog=(
            "Examples:\n"
            "  pulgon-irreps-tables -p POSCAR -q 0.0 -s characters\n"
            "  pulgon-irreps-tables -p POSCAR -q 0.25 -r -s irreps_q025\n"
            "\n"
            "Notes:\n"
            "  qpoint_z is the reduced q coordinate along the periodic z "
            "direction.\n"
            "  Output is saved as an .npz file; for example, -s characters "
            "writes characters.npz.\n"
            "  Without -r, the file stores character tables and irrep labels "
            "only.\n"
            "  With -r, representation matrices D_irrep_* are also stored."
        ),
        formatter_class=RawDescriptionDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--POSCAR",
        help="Input structure file in a format readable by ASE.",
    )
    parser.add_argument(
        "-q",
        "--qpoint_z",
        type=float,
        default=0.0,
        help=(
            "Reduced q coordinate along the periodic z direction; internally "
            "converted to q = qpoint_z * 2*pi/a."
        ),
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Optional numerical tolerance. If omitted, symmetry detection "
            "uses 1e-2 and character evaluation uses 1e-8; if set, the same "
            "value is used for both."
        ),
    )
    parser.add_argument(
        "-s",
        "--savename_chara",
        default="characters",
        help=(
            "Output base filename for the .npz file; the saved file is "
            "<name>.npz."
        ),
    )
    parser.add_argument(
        "-r",
        "--enable_rep_matrix",
        action="store_true",
        help=(
            "Also save irreducible representation matrices as D_irrep_* "
            "arrays in the output .npz file."
        ),
    )

    args = parser.parse_args()
    st_name = args.POSCAR
    qpoint_z = args.qpoint_z
    symprec = 1e-8 if args.tolerance is None else args.tolerance
    chara_filename = args.savename_chara
    enable_rep_matrix = args.enable_rep_matrix

    atom = read(st_name)
    symmetry_tolerance = 1e-2 if args.tolerance is None else args.tolerance

    (
        atom_center,
        family,
        nrot,
        aL,
        ops_car_sym,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(atom, tolerance=symmetry_tolerance)
    qp_normalized = qpoint_z / aL * 2 * np.pi
    DictParams: Dict[str, object] = {
        "qpoints": qp_normalized,
        "nrot": nrot,
        "order": order_ops,
        "family": family,
        "a": aL,
        **gen_angles,
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
        representation_mat_dict: Dict[str, np.ndarray] = {}
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
