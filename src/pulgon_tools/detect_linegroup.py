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
from typing import Tuple

from ase import Atoms
from ase.io.vasp import read_vasp

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer
from pulgon_tools.line_group_table import get_family_num_from_sym_symbol
from pulgon_tools.utils import find_axis_center_of_nanotube


def detect_linegroup(
    atom: Atoms,
    tolerance: float = 1e-2,
) -> Tuple[str, str, int]:
    """Detect the line-group symbols and family number.

    Args:
        atom: line-group structure.
        tolerance: distance tolerance for symmetry detection.

    Returns:
        tuple of (generalized translational group Z, axial point group P,
        line-group family number).
    """
    cyclic = CyclicGroupAnalyzer(atom, tolerance=tolerance)
    atom_center = find_axis_center_of_nanotube(cyclic._primitive)

    point_group = LineGroupAnalyzer(atom_center, tolerance=tolerance)
    cyclic_groups, _ = cyclic.get_cyclic_group()
    trans_sym = cyclic_groups[0]
    rota_sym = point_group.sch_symbol
    family = get_family_num_from_sym_symbol(trans_sym, rota_sym)

    return trans_sym, rota_sym, family


def main() -> None:
    """CLI entry point for detecting line-group symbols."""
    parser = argparse.ArgumentParser(
        description="Detect the line group of a nanostructure"
    )
    parser.add_argument(
        "-p",
        "--POSCAR",
        required=True,
        help="path to the file from which coordinates will be read",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        default=1e-2,
        type=float,
        help="Tolerance for atomic positions",
    )

    args = parser.parse_args()

    atom = read_vasp(args.POSCAR)
    z_sym, p_sym, family = detect_linegroup(atom, tolerance=args.tolerance)

    print("generalized translational group Z:", z_sym)
    print("axial point group P:", p_sym)
    print("line group family number:", family)


if __name__ == "__main__":
    main()
