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


from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from ase.io.vasp import write_vasp

from pulgon_tools.structures_sym_based import generate_line_group_structure
from pulgon_tools.utils import Cn, S2n, U, U_d, dimino, sigmaH, sigmaV

FamilyInput = Tuple[np.ndarray, np.ndarray, np.ndarray, dict, str]


def input1() -> FamilyInput:
    """(Cq|f),Cn"""
    motif = np.array([[2.0, 0.21, 0.0], [2.7, 0.83, 0.0]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3)])
    cyclic = {"T_Q": [5, 1.2]}
    st_name = "family_01.vasp"
    return motif, symbols, generators, cyclic, st_name


def input2() -> FamilyInput:
    """(I|q),S2n"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([S2n(3)])
    cyclic = {"T_Q": [1, 3.0]}
    st_name = "family_02.vasp"
    return motif, symbols, generators, cyclic, st_name


def input3() -> FamilyInput:
    """(I|q),Cn,sigmaH"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3), sigmaH()])
    cyclic = {"T_Q": [1, 3.0]}
    st_name = "family_03.vasp"
    return motif, symbols, generators, cyclic, st_name


def input4() -> FamilyInput:
    """(C2n|f/2),Cn,sigmaH"""
    motif = np.array([[2.0, 0.21, 0.13], [2.7, 0.83, 0.41]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(2), sigmaH()])
    cyclic = {"T_Q": [4, 1.2]}
    st_name = "family_04.vasp"
    return motif, symbols, generators, cyclic, st_name


def input5() -> FamilyInput:
    """(Cq|f),Cn,U"""
    motif = np.array([[2.0, 0.21, 0.0], [2.7, 0.83, 0.0]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3), U()])
    cyclic = {"T_Q": [5, 1.2]}
    st_name = "family_05.vasp"
    return motif, symbols, generators, cyclic, st_name


def input6() -> FamilyInput:
    """(I|a),Cn,sigmaV"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3), sigmaV()])
    cyclic = {"T_Q": [1, 3.0]}
    st_name = "family_06.vasp"
    return motif, symbols, generators, cyclic, st_name


def input7() -> FamilyInput:
    """(sigmaV|a/2),Cn"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3)])
    cyclic = {"T_V": 1.5}
    st_name = "family_07.vasp"
    return motif, symbols, generators, cyclic, st_name


def input8() -> FamilyInput:
    """(C2n|a/2),Cn,sigmaV"""
    motif = np.array([[2.0, 0.21, 0.13], [2.7, 0.83, 0.41]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(2), sigmaV()])
    cyclic = {"T_Q": [4, 1.2]}
    st_name = "family_08.vasp"
    return motif, symbols, generators, cyclic, st_name


def input9() -> FamilyInput:
    """(I|a),Cn,Ud,sigmaV"""
    motif = np.array([[2.0, 0.21, 0.13], [2.7, 0.83, 0.41]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(2), U_d(np.pi / 4), sigmaV()])
    cyclic = {"T_Q": [1, 3.0]}
    st_name = "family_09.vasp"
    return motif, symbols, generators, cyclic, st_name


def input10() -> FamilyInput:
    """(sigmaV|a/2),S2n"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([S2n(3)])
    cyclic = {"T_V": 1.5}
    st_name = "family_10.vasp"
    return motif, symbols, generators, cyclic, st_name


def input11() -> FamilyInput:
    """(I|a),Cn,sigmaV"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3), U(), sigmaH()])
    cyclic = {"T_Q": [1, 3.0]}
    st_name = "family_11.vasp"
    return motif, symbols, generators, cyclic, st_name


def input12() -> FamilyInput:
    """(sigmaV|a),Cn,U,sigmaV"""
    motif = np.array([[2.0, 0.21, 0.4], [2.7, 0.83, 0.9]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(3), sigmaH()])
    cyclic = {"T_V": 1.5}
    st_name = "family_12.vasp"
    return motif, symbols, generators, cyclic, st_name


def input13() -> FamilyInput:
    """(C2n|a/2),Cn,U,sigmaV"""
    motif = np.array([[2.0, 0.21, 0.13], [2.7, 0.83, 0.41]])
    symbols = np.array(["C", "N"])
    generators = np.array([Cn(2), U(), sigmaV()])
    cyclic = {"T_Q": [4, 1.2]}
    st_name = "family_13.vasp"
    return motif, symbols, generators, cyclic, st_name


def _build_monomer(
    pos_cylin: np.ndarray, symbols: np.ndarray, generators: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pos_cylin = np.atleast_2d(pos_cylin)
    if len(symbols) != len(pos_cylin):
        raise ValueError("The number of symbols must match motif atoms.")

    pos = np.array(
        [
            pos_cylin[:, 0] * np.cos(pos_cylin[:, 1]),
            pos_cylin[:, 0] * np.sin(pos_cylin[:, 1]),
            pos_cylin[:, 2],
        ]
    ).T
    rot_sym = dimino(generators, symec=4)
    monomer_pos, monomer_symbols = [], []
    for sym in rot_sym:
        monomer_pos.extend([np.dot(sym, line) for line in pos])
        monomer_symbols.extend(symbols)
    monomer_pos = np.array(monomer_pos)
    monomer_symbols = np.array(monomer_symbols)
    return monomer_pos, monomer_symbols


def generate_structure(
    input_func: Callable[[], FamilyInput],
    output_dir: Path,
) -> Path:
    pos_cylin, symbols, generators, cg, st_name = input_func()
    monomer_pos, monomer_symbols = _build_monomer(
        pos_cylin, symbols, generators
    )
    st = generate_line_group_structure(
        monomer_pos, monomer_symbols, cg, symec=4
    )
    output_path = output_dir / st_name
    write_vasp(output_path, st, direct=True, sort=True)
    return output_path


def main() -> None:
    output_dir = Path("examples/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_funcs = [
        input1,
        input2,
        input3,
        input4,
        input5,
        input6,
        input7,
        input8,
        input9,
        input10,
        input11,
        input12,
        input13,
    ]
    for input_func in input_funcs:
        output_path = generate_structure(input_func, output_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
