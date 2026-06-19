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
import copy
import re
from typing import Sequence, Union

import ase
import numpy as np
from ase import Atoms
from ase.io.vasp import write_vasp

from pulgon_tools.cli import (
    RawDescriptionDefaultsHelpFormatter,
    parse_motif_groups,
    parse_symbols,
)
from pulgon_tools.utils import (
    Cn,
    S2n,
    U,
    U_d,
    dimino,
    sigmaH,
    sigmaV,
    sortrows,
)

_GENERATOR_PATTERNS = {
    "Cn": Cn,
    "S2n": S2n,
    "U_d": U_d,
}

_GENERATOR_NOARG = {
    "sigmaV()": sigmaV,
    "sigmaH()": sigmaH,
    "U()": U,
}

_SUPPORTED_GENERATORS = (
    "Cn(number), S2n(number), U_d(angle), sigmaV(), sigmaH(), U()"
)

DEFAULT_MOTIF = [[3, 0, 0], [2.2, 0.2618, 0]]
DEFAULT_SYMBOLS = ("C", "N")
DEFAULT_GENERATORS = ["Cn(8)"]
DEFAULT_CYCLIC = ["T_Q", "3", "1.6"]


def _parse_cyclic_group(value: Sequence[str]) -> dict:
    key = value[0]
    if key == "T_Q":
        if len(value) != 3:
            raise ValueError("--cyclic T_Q must be given as: T_Q Q f.")
        q, f = float(value[1]), float(value[2])
        if q <= 0 or f <= 0:
            raise ValueError(
                "--cyclic T_Q parameters Q and f must be positive."
            )
        return {"T_Q": [q, f]}

    if key == "T_V":
        if len(value) != 2:
            raise ValueError("--cyclic T_V must be given as: T_V f.")
        f = float(value[1])
        if f <= 0:
            raise ValueError("--cyclic T_V value f must be positive.")
        return {"T_V": f}

    raise ValueError("--cyclic must start with 'T_Q' or 'T_V'.")


def _parse_generator(generator: str) -> np.ndarray:
    if generator in _GENERATOR_NOARG:
        return _GENERATOR_NOARG[generator]()

    match = re.fullmatch(r"(Cn|S2n|U_d)\(([-+]?\d+(?:\.\d+)?)\)", generator)
    if match:
        name, parameter = match.groups()
        value = float(parameter)
        if name in {"Cn", "S2n"} and value <= 0:
            raise ValueError(f"{name} generator parameter must be positive.")
        return _GENERATOR_PATTERNS[name](value)

    raise ValueError(
        f"Unsupported generator '{generator}'. Supported generators: "
        f"{_SUPPORTED_GENERATORS}."
    )


def T_Q(
    Q: Union[float, int], f: Union[float, int], pos: np.ndarray
) -> np.ndarray:
    """

    Args:
        Q: rotate 2*pi/Q
        f: movement distance along z axis
        pos: monomer positions

    Returns: atom positions after rotation by 2*pi/Q
        and translation by f along z

    """
    if pos.ndim == 2:
        result = np.dot(Cn(Q), pos.T).T
    else:
        print("error with dim")
        return pos
    result[:, 2] = result[:, 2] + f
    return result


def T_v(f: Union[float, int], pos: np.ndarray) -> np.ndarray:
    """

    Args:
        f: movement distance along z axis
        pos: monomer positions

    Returns: atom positions after mirror reflection
        (sigmaV) and translation by f along z

    """
    if pos.ndim == 2:
        result = np.dot(sigmaV(), pos.T).T
    else:
        print("error with dim")
        return pos
    result[:, 2] = result[:, 2] + f
    return result


def change_center(st1: ase.atoms.Atoms) -> ase.atoms.Atoms:
    """

    Args:
        st1: an ase.atom structure

    Returns: an ase.atom structure with z axis located in the cell center

    """
    st1_pos = st1.get_scaled_positions()
    st2_pos = st1_pos[:, :2] + 0.5
    tmp = np.modf(st2_pos)[0]
    tmp1 = st1_pos[:, 2]
    tmp1 = tmp1.reshape(tmp1.shape[0], 1)

    st2 = copy.deepcopy(st1)
    st2.positions = np.dot(np.hstack((tmp, tmp1)), st2.cell)
    return st2


def generate_line_group_structure(
    monomer_pos: np.ndarray,
    monomer_symbols: Union[list, np.ndarray],
    cyclic_group: dict,
    symprec: int = 4,
) -> ase.atoms.Atoms:
    """
    Args:
        monomer_pos: the positions of monomer
        monomer_symbols: atomic symbols of monomer
        cyclic_group: the generalized translation group
        symprec: number of decimal places for rounding in deduplication

    Returns:
        the final structure after all symmetry operations
    """

    all_pos = copy.deepcopy(monomer_pos)
    all_symbols = list(monomer_symbols)

    if list(cyclic_group.keys())[0] == "T_Q":
        Q = cyclic_group["T_Q"][0]
        f = cyclic_group["T_Q"][1]

        tmp_monomer_pos = monomer_pos.copy()
        for ii in range(np.ceil(Q).astype(np.int32)):

            tmp_monomer_pos = T_Q(Q, f, tmp_monomer_pos)

            all_pos = np.vstack((all_pos, tmp_monomer_pos))
            all_symbols.extend(monomer_symbols)

            judge = np.sum(
                (
                    sortrows(np.round(monomer_pos[:, :2], symprec))
                    - sortrows(np.round(tmp_monomer_pos[:, :2], symprec))
                )
                ** 2
            )

            if judge < 0.1:
                Q = ii + 1
                break

        # Remove exact cartesian duplicates
        pos_round = np.round(all_pos, symprec)
        _, unique_idx = np.unique(pos_round, axis=0, return_index=True)
        all_pos = all_pos[unique_idx]
        all_symbols = [all_symbols[i] for i in unique_idx]

        A = Q * f

    elif list(cyclic_group.keys())[0] == "T_V":
        f = cyclic_group["T_V"]
        for ii in range(2):
            new_pos = T_v(f, all_pos)
            all_pos = np.vstack((all_pos, new_pos))
            all_symbols.extend(all_symbols[: len(new_pos)])

        # Remove exact cartesian duplicates
        pos_round = np.round(all_pos, symprec)
        _, unique_idx = np.unique(pos_round, axis=0, return_index=True)
        all_pos = all_pos[unique_idx]
        all_symbols = [all_symbols[i] for i in unique_idx]

        A = 2 * f
    else:
        print("Invalid cyclic_group key, expected 'T_Q' or 'T_V'")

    p0 = np.max(np.sqrt(all_pos[:, 0] ** 2 + all_pos[:, 1] ** 2))
    cell = np.array([[p0 * 3, 0, 0], [0, p0 * 3, 0], [0, 0, A]])
    st1 = Atoms(symbols=all_symbols, positions=all_pos, cell=cell)

    st2 = change_center(st1)

    # Remove periodic duplicates along z using scaled positions
    scaled = st2.get_scaled_positions()
    # Fold into [0, 1) and round; snap near-boundary values to 0
    scaled_folded = np.round(scaled % 1, symprec)
    scaled_folded[scaled_folded >= 1.0] = 0.0
    pos_uni, idx = np.unique(scaled_folded, return_index=True, axis=0)
    symbols = st2.symbols[idx]

    st3 = Atoms(
        symbols=symbols,
        scaled_positions=pos_uni,
        cell=cell,
        pbc=[False, False, True],
    )

    return st3


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 1D line-group structure from a cylindrical motif, "
            "point-group generators, and a generalized translation."
        ),
        epilog=(
            "Examples:\n"
            "  pulgon-generate-structures-sym_based "
            "-m 3 0.13 0.6 "
            "2.2 0.13 0.8 "
            "-b Mo S "
            '-g "Cn(6)" "sigmaV()" '
            "-c T_Q 6 1.5 "
            "-s poscar.vasp\n\n"
            "Notes:\n"
            "  - Motif coordinates are [r, phi, z] in cylindrical "
            "coordinates; provide one group of three values per motif atom.\n"
            "  - Supported generators: Cn(number), S2n(number), U_d(angle), "
            "sigmaV(), sigmaH(), U().\n"
            "  - Generalized translations are T_Q Q f or T_V f."
        ),
        formatter_class=RawDescriptionDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--motif",
        nargs="+",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Motif coordinates as r phi z groups; the number of values must "
            "be a multiple of three."
        ),
    )

    parser.add_argument(
        "-g",
        "--generators",
        nargs="+",
        default=DEFAULT_GENERATORS,
        help=(
            "Point-group generators for the motif. Supported: "
            "Cn(number), S2n(number), U_d(angle), sigmaV(), sigmaH(), U()."
        ),
    )

    parser.add_argument(
        "-c",
        "--cyclic",
        nargs="+",
        default=DEFAULT_CYCLIC,
        metavar="ARG",
        help=(
            "Generalized translation group: T_Q Q f for a screw operation "
            "or T_V f for a glide operation; f is the z translation."
        ),
    )

    parser.add_argument(
        "-s",
        "--st_name",
        type=str,
        default="structure.vasp",
        help="Output VASP/POSCAR filename.",
    )
    parser.add_argument(
        "-b",
        "--symbol",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Atomic symbols for the motif atoms, in the same order as -m.",
    )

    args = parser.parse_args()
    try:
        symbols = parse_symbols(args.symbol)
        pos_cylin = parse_motif_groups(args.motif, DEFAULT_MOTIF)
        generators = np.array(
            [_parse_generator(tmp) for tmp in args.generators]
        )
        cg = _parse_cyclic_group(args.cyclic)
    except (SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    if len(symbols) != len(pos_cylin):
        parser.error(
            "--symbol must provide exactly one symbol for each -m motif atom."
        )

    if pos_cylin.ndim == 1:
        pos = np.array(
            [
                pos_cylin[0] * np.cos(pos_cylin[1]),
                pos_cylin[0] * np.sin(pos_cylin[1]),
                pos_cylin[2],
            ]
        )
    else:
        pos = np.array(
            [
                pos_cylin[:, 0] * np.cos(pos_cylin[:, 1]),
                pos_cylin[:, 0] * np.sin(pos_cylin[:, 1]),
                pos_cylin[:, 2],
            ]
        )
        pos = pos.T
    st_name = args.st_name

    rot_sym = dimino(generators, symprec=3)

    monomer_pos, monomer_symbols = [], []
    for sym in rot_sym:
        if pos.ndim == 1:
            monomer_pos.append(np.dot(sym, pos.reshape(pos.shape[0], 1)).T[0])
            monomer_symbols.extend(symbols)
        else:
            monomer_pos.extend([np.dot(sym, line) for line in pos])
            monomer_symbols.extend(symbols)

    monomer_pos = np.array(monomer_pos)
    monomer_symbols = np.array(monomer_symbols)

    st = generate_line_group_structure(
        monomer_pos, monomer_symbols, cg, symprec=3
    )
    write_vasp("%s" % st_name, st, direct=True, sort=True)
    print(f"Successfully generated {st_name}.")


if __name__ == "__main__":
    main()
