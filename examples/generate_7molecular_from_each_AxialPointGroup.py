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

from typing import Tuple

import numpy as np
from ase.io.vasp import write_vasp

from pulgon_tools.structures_sym_based import generate_line_group_structure
from pulgon_tools.utils import Cn, S2n, U, U_d, dimino, sigmaH, sigmaV


def input1() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Cn"""
    motif = np.array([3, 0, 1.5])
    generators = np.array([Cn(6)])
    cyclic = {"T_Q": [1, 3]}
    st_name = "m1.vasp"
    return motif, generators, cyclic, st_name


def input2() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """S2n"""
    motif = np.array([3, 0, 1.5])
    generators = np.array([S2n(6)])
    cyclic = {"T_Q": [1, 4]}
    st_name = "m2.vasp"
    return motif, generators, cyclic, st_name


def input3() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Cnh"""
    motif = np.array([3, 0, 1.5])
    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_Q": [1, 4]}
    st_name = "m3.vasp"
    return motif, generators, cyclic, st_name


def input4() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Dn"""
    motif = np.array([3, np.pi / 24, 1.5])
    generators = np.array([Cn(6), U()])
    cyclic = {"T_Q": [1, 4]}
    st_name = "m4.vasp"
    return motif, generators, cyclic, st_name


def input5() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Cnv"""
    motif = np.array([3, np.pi / 24, 1.5])
    generators = np.array([Cn(6), sigmaV()])
    cyclic = {"T_Q": [1, 3]}
    st_name = "m5.vasp"
    return motif, generators, cyclic, st_name


def input6() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Dnd"""
    motif = np.array([3, np.pi / 24, 1.5])
    generators = np.array([S2n(6), U_d(np.pi / 12)])
    cyclic = {"T_Q": [1, 4]}
    st_name = "m6.vasp"
    return motif, generators, cyclic, st_name


def input7() -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Dnh"""
    motif = np.array([3, np.pi / 24, 1.5])
    generators = np.array([Cn(6), U(), sigmaV()])
    cyclic = {"T_Q": [1, 4]}
    st_name = "m7.vasp"
    return motif, generators, cyclic, st_name


def main() -> None:
    pos_cylin, generators, cg, st_name = input7()

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
    rot_sym = dimino(generators, symec=4)
    monomer_pos = []
    for sym in rot_sym:
        if pos.ndim == 1:
            monomer_pos.append(np.dot(sym, pos.reshape(pos.shape[0], 1)).T[0])
        else:
            monomer_pos.extend([np.dot(sym, line) for line in pos])
    monomer_pos = np.array(monomer_pos)

    st = generate_line_group_structure(monomer_pos, cg)
    write_vasp("%s" % st_name, st, direct=True, sort=True)


if __name__ == "__main__":
    main()
