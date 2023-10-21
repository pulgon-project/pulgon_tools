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

import itertools
import json

import numpy as np
from ase import Atoms
from ipdb import set_trace

from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer


def sortrows(a: np.ndarray) -> np.ndarray:
    """
    :param a:
    :return: Compare each row in ascending order
    """
    return a[np.lexsort(np.rot90(a))]


def refine_cell(
    scale_pos: np.ndarray, numbers: np.ndarray, symprec: int = 4
) -> [np.ndarray, np.ndarray]:
    """refine the scale position between 0-1, and remove duplicates

    Args:
        scale_pos: scale position of the structure
        numbers: atom_type
        symprec: system precise

    Returns: scale position after refine, the correspond atom type

    """
    if scale_pos.ndim == 1:
        scale_pos = np.modf(scale_pos)[0]
        scale_pos[scale_pos < 0] = scale_pos[scale_pos < 0] + 1
        pos = np.round(scale_pos, symprec)
    else:
        scale_pos = np.modf(scale_pos)[0]
        scale_pos[scale_pos < 0] = scale_pos[scale_pos < 0] + 1
        # set_trace()
        scale_pos = np.round(scale_pos, symprec)

        pos, index = np.unique(scale_pos, axis=0, return_index=True)
        numbers = numbers[index]
    return pos, numbers


def frac_range(
    start: float,
    end: float,
    left: bool = True,
    right: bool = True,
    symprec: float = 0.01,
) -> list:
    """return the integer within the specified range

    Args:
        start: left boundary
        end: right boundary
        left: False mean delete the left boundary element if it is an integer
        right: False mean delete the right boundary element if it is an integer
        symprec: system precise

    Returns:

    """
    close = list(
        range(
            np.ceil(start).astype(np.int32), np.floor(end).astype(np.int32) + 1
        )
    )
    if left == False:
        if close[0] - start < symprec:
            close.pop(0)  # delete the left boundary
    if right == False:
        if close[-1] - end < symprec:
            close.pop()  # delete the right boundary
    return close


def get_num_of_decimal(num: float) -> int:
    return len(np.format_float_positional(num).split(".")[1])


def get_symcell(monomer: Atoms) -> Atoms:
    """based on the point group symmetry of monomer, return the symcell

    Args:
        monomer:

    Returns: symcell

    """
    apg = LineGroupAnalyzer(monomer)
    equ = list(apg.get_equivalent_atoms()["eq_sets"].keys())
    # sym = apg.get_symmetry_operations()
    return monomer[equ]


def get_perms(atoms, cyclic_group_ops, point_group_ops, symprec=1e-3):

    combs = list(itertools.product(point_group_ops, cyclic_group_ops))

    coords = atoms.get_scaled_positions()
    for ii, op in enumerate(combs):
        op1 = op[0]
        op2 = op[1]

        for coord in coords:
            tmp = op1.operate(coord)
            tmp, _ = refine_cell(tmp, 1)

            idx = np.argmin(np.linalg.norm(tmp - coords, axis=1))
            set_trace()

    # perms = np.zeros((np.shape(trans)[0], len(atoms.numbers)))
    origin_positions, numbers = refine_cell(
        atoms.get_scaled_positions(), atoms.numbers
    )
    for ix, rot in enumerate(point_group_ops):
        for ix, rot in enumerate(cyclic_group_ops):
            for iy, o_pos in enumerate(origin_positions):
                new_pos = np.dot(rot, o_pos.T) + trans[ix]
                new_pos = np.mod(new_pos, 1)
                new_pos, new_numbers = refine_cell(new_pos, numbers)
                idx = np.argmin(
                    np.linalg.norm(new_pos - origin_positions, axis=1)
                )
                perms[ix, iy] = idx
    perms_table = np.unique(perms, axis=0)
    return perms_table
