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
from typing import Union

import ase
import numpy as np
from ase import Atoms
from ase.io.vasp import write_vasp

from pulgon_tools.utils import Cn, dimino, sigmaV, sortrows


def T_Q(
    Q: Union[float, int], f: Union[float, int], pos: np.ndarray
) -> np.ndarray:
    """

    Args:
        Q: rotate 2*pi/n
        f: movement distance along z axis
        pos: monomer positions

    Returns: the positions of all atoms

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

    Returns: the positions of all atoms

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
    monomer_symbols,
    cyclic_group: dict,
    symec: int = 4,
) -> ase.atoms.Atoms:
    """
    Args:
        monomer_pos: the positions of monomer
        monomer_symbols: atomic symbols of monomer
        cyclic_group: the generalized translation group

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
                    sortrows(np.round(monomer_pos[:, :2], symec))
                    - sortrows(np.round(tmp_monomer_pos[:, :2], symec))
                )
                ** 2
            )

            if judge < 0.1:
                Q = ii + 1
                break

        # Remove exact cartesian duplicates
        pos_round = np.round(all_pos, symec)
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
        pos_round = np.round(all_pos, symec)
        _, unique_idx = np.unique(pos_round, axis=0, return_index=True)
        all_pos = all_pos[unique_idx]
        all_symbols = [all_symbols[i] for i in unique_idx]

        A = 2 * f
    else:
        print("A error input about cyclic_group")

    p0 = np.max(np.sqrt(all_pos[:, 0] ** 2 + all_pos[:, 1] ** 2))
    cell = np.array([[p0 * 3, 0, 0], [0, p0 * 3, 0], [0, 0, A]])
    st1 = Atoms(symbols=all_symbols, positions=all_pos, cell=cell)

    st2 = change_center(st1)

    # Remove periodic duplicates along z using scaled positions
    scaled = st2.get_scaled_positions()
    # Fold into [0, 1) and round; snap near-boundary values to 0
    scaled_folded = np.round(scaled % 1, symec)
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
        description="generating line group structure by symmetry-based approach"
    )

    parser.add_argument(
        "-m",
        "--motif",
        default=[[3, np.pi / 24, 0.6], [2.2, np.pi / 24, 0.8]],
        help="the Cylindrical coordinates of initial atom position",
    )

    parser.add_argument(
        "-g",
        "--generators",
        default=["Cn(6)", "sigmaV()"],
        help="the point group generator of monomer",
    )

    parser.add_argument(
        "-c",
        "--cyclic",
        default={"T_Q": [6, 1.5]},
        help="The generalized translation group. For T_Q the first parameter is Q and the second parameter is f."
        " For T_V the parameter is f",
    )

    parser.add_argument(
        "-s",
        "--st_name",
        type=str,
        default="poscar.vasp",
        help="the saved file name",
    )
    parser.add_argument(
        "-b",
        "--symbol",
        default=("Mo", "S"),
    )

    args = parser.parse_args()
    symbols = eval(args.symbol)

    pos_cylin = np.array(eval(args.motif))
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
    generators = np.array([eval(tmp) for tmp in eval(args.generators)])
    cg = eval(args.cyclic)
    st_name = args.st_name

    rot_sym = dimino(generators, symec=3)

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
        monomer_pos, monomer_symbols, cg, symec=3
    )
    write_vasp("%s" % st_name, st, direct=True, sort=True)


if __name__ == "__main__":
    main()
