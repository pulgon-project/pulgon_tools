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
from ase.io import read
from ase.io.formats import UnknownFileTypeError

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer
from pulgon_tools.line_group_table import get_family_num_from_sym_symbol
from pulgon_tools.utils import find_axis_center_of_nanotube


def _read_structure(filename: str) -> Atoms:
    """Read a structure with ASE, preserving extensionless POSCAR support."""
    try:
        return read(filename)
    except UnknownFileTypeError:
        return read(filename, format="vasp")


def detect_linegroup(
    atom: Atoms,
    tolerance: float = 1e-2,
    layer_tolerance: float = 0.05,
) -> Tuple[str, str, int]:
    """Detect the line-group symbols and family number.

    Args:
        atom: line-group structure.
        tolerance: distance tolerance for symmetry detection.
        layer_tolerance: fractional tolerance for cyclic-group z-layer and
            monomer translation candidate detection.

    Returns:
        tuple of (generalized translational group Z, axial point group P,
        line-group family number).
    """
    cyclic = CyclicGroupAnalyzer(
        atom,
        tolerance=tolerance,
        layer_tolerance=layer_tolerance,
    )
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
        description=(
            "Detect the generalized translational group Z, axial point "
            "group P, and line-group family number of a 1D periodic structure."
        ),
        epilog=(
            "Examples:\n"
            "  pulgon-detect-linegroup -p POSCAR -t 1e-2 "
            "--layer-tolerance 0.05\n\n"
            "Notes:\n"
            "  - The periodic axis is expected to be along the Cartesian "
            "z direction.\n"
            "  - Z is the generalized translational group; P is the axial "
            "point group.\n"
            "  - tolerance matches atomic positions; layer-tolerance groups "
            "z-layers into monomer translation candidates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--POSCAR",
        required=True,
        help="Input structure file in a format readable by ASE.",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        default=1e-2,
        type=float,
        help=(
            "Distance tolerance in Angstrom for matching transformed atoms "
            "to existing atoms during symmetry detection."
        ),
    )
    parser.add_argument(
        "-d",
        "--layer-tolerance",
        default=0.05,
        type=float,
        help=(
            "Fractional z-coordinate tolerance for grouping atomic layers "
            "and selecting monomer translation candidates."
        ),
    )

    args = parser.parse_args()

    atom = _read_structure(args.POSCAR)
    z_sym, p_sym, family = detect_linegroup(
        atom,
        tolerance=args.tolerance,
        layer_tolerance=args.layer_tolerance,
    )

    print("generalized translational group Z:", z_sym)
    print("axial point group P:", p_sym)
    print("line group family number:", family)


if __name__ == "__main__":
    main()
