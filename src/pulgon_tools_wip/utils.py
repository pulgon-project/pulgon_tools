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

import copy
import itertools
import json
import logging

import numpy as np
from ase import Atoms
from ipdb import set_trace
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import find_in_coord_list

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
        pos = scale_pos
        # pos = np.round(scale_pos, symprec)
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


def get_perms(atoms, cyclic_group_ops, point_group_ops, symprec=1e-2):
    """get the permutation table from symmetry operations

    Args:
        atoms:
        cyclic_group_ops:
        point_group_ops:
        symprec:

    Returns: permutation table
             rotation matrix (RM) = RM from point group @ RM from cyclic group
    """
    combs = list(itertools.product(point_group_ops, cyclic_group_ops))
    coords_car = atoms.positions
    coords_scaled = atoms.get_scaled_positions()
    coords_car_center = (
        atoms.get_scaled_positions() - [0.5, 0.5, 0.5]
    ) @ atoms.cell

    perms, rotation_matrix, translation_vector = [], [], []
    for ii, op in enumerate(combs):
        tmp_perm = np.ones((1, len(atoms.numbers)))[0]
        op1, op2 = op
        rotation_matrix.append(op1.rotation_matrix @ op2.rotation_matrix)
        translation_vector.append(
            op1.translation_vector + op2.translation_vector
        )

        for jj, site in enumerate(atoms):
            pos = (site.scaled_position - [0.5, 0.5, 0.5]) @ atoms.cell

            tmp = op1.operate(pos)
            idx1 = find_in_coord_list(coords_car_center, tmp, symprec)

            tmp1 = op2.operate(coords_car[idx1.item()])
            tmp1 = np.remainder(tmp1 @ np.linalg.inv(atoms.cell), [1, 1, 1])
            idx2 = find_in_coord_list(coords_scaled, tmp1, symprec)

            if idx2.size == 0:
                logging.ERROR("tolerance exceed while calculate perms")
            tmp_perm[jj] = idx2
        perms.append(tmp_perm)
    perms_table, itp = np.unique(perms, axis=0, return_index=True)
    perms_table = perms_table.astype(np.int32)

    rotation_matrix = np.array(rotation_matrix)[itp]
    translation_vector = np.array(translation_vector)[itp]
    sym_operations = [
        SymmOp.from_rotation_and_translation(
            rotation_matrix[ii], translation_vector[ii]
        )
        for ii in range(len(itp))
    ]
    return perms_table, sym_operations


def fast_orth(A, maxrank):
    """Reimplementation of scipy.linalg.orth() which takes only the vectors with
    values almost equal to the maximum, and returns at most maxrank vectors.
    """
    u, s, vh = la.interpolative.svd(A, maxrank)
    reference = s[0]
    for i in range(s.size):
        if abs(reference - s[i]) > 0.05 * reference:
            return u[:, :i]
    return u


def affine_matrix_op(af1, af2):
    ro = af1[:3, :3] @ af2[:3, :3]
    tran = np.remainder(af1[:3, 3] + af2[:3, 3], [1, 1, 1])
    af = np.eye(4)
    af[:3, :3] = ro
    af[:3, 3] = tran
    return af


def dimino_affine_matrix_and_character(
    generators: np.ndarray, character, symec: float = 0.001
) -> np.ndarray:
    """

    Args:
        generators: the generators of point group
        symec: system precision

    Returns: all the group elements and correspond character

    """
    e_in = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    if character[0].ndim == 0:
        L_chara = np.array([np.complex128(1)])
        C_chara = np.array([np.complex128(1)])
    else:
        L_chara = np.array([np.complex128(1) * np.eye(character[0].shape[0])])
        C_chara = np.array([np.complex128(1) * np.eye(character[0].shape[0])])

    G = generators
    g, g1 = G[0].copy(), G[0].copy()
    g_chara, g1_chara = character[0].copy(), character[0].copy()
    L = np.array([e_in])
    while not ((g - e_in) < symec).all():
        L = np.vstack((L, [g]))
        L_chara = np.vstack((L_chara, [g_chara]))

        # g = np.dot(g, g1)
        g = affine_matrix_op(g, g1)
        g_chara = np.dot(g_chara, g1_chara)

    for ii in range(len(G)):
        C = np.array([e_in])
        L1 = L.copy()
        L1_chara = L_chara.copy()

        more = True
        while more:
            more = False
            for jj, g in enumerate(list(C)):
                g_chara = C_chara[jj]

                for kk, ss in enumerate(G[: ii + 1]):
                    ss_chara = character[kk]
                    sg = affine_matrix_op(ss, g)
                    sg_chara = np.dot(ss_chara, g_chara)

                    itp = ((sg - L).sum(axis=1).sum(axis=1) < 0.001).any()
                    if not itp:
                        if C.ndim == 3:
                            C = np.vstack((C, [sg]))
                            C_chara = np.vstack((C_chara, [sg_chara]))
                        else:
                            C = np.array((C, sg))
                            C_chara = np.array((C_chara, sg_chara))

                        if L.ndim == 3:
                            L = np.vstack(
                                (
                                    L,
                                    np.array(
                                        [affine_matrix_op(sg, t) for t in L1]
                                    ),
                                )
                            )
                            tmp = np.array(
                                [
                                    np.dot(sg_chara, t_chara)
                                    for t_chara in L1_chara
                                ]
                            )
                            L_chara = np.vstack((L_chara, tmp))
                        else:
                            L = np.array(
                                L,
                                np.array(
                                    [affine_matrix_op(sg, t) for t in L1]
                                ),
                            )
                        more = True
    L_chara_trace = []
    for lc in L_chara:
        if lc.ndim == 1:
            L_chara_trace.append(lc)
        else:
            L_chara_trace.append(np.trace(lc))
    return L, np.array(L_chara_trace)


def dimino_affine_matrix_and_subsquent(
    generators: np.ndarray, symec: float = 0.001
) -> np.ndarray:
    """

    Args:
        generators: the generators of point group
        symec: system precision

    Returns: all the group elements and correspond character

    """
    e_in = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    G = generators
    g, g1 = G[0].copy(), G[0].copy()
    g_subs, g1_subs = [1], [1]
    L = np.array([e_in])
    L_subs = [[0]]

    while not ((g - e_in) < symec).all():
        L = np.vstack((L, [g]))
        L_subs.append(g_subs.copy())

        g = affine_matrix_op(g, g1)
        g_subs = g_subs + g1_subs
    for ii in range(len(G)):
        C = np.array([e_in])
        C_subs = [[0]]

        L1 = L.copy()
        L1_subs = L_subs.copy()
        more = True
        while more:
            more = False
            for jj, g in enumerate(list(C)):
                g_subs = C_subs[jj]

                for kk, ss in enumerate(G[: ii + 1]):
                    ss_subs = [kk + 1]

                    sg = affine_matrix_op(ss, g)
                    sg_subs = ss_subs + g_subs

                    itp = ((sg - L).sum(axis=1).sum(axis=1) < 0.001).any()
                    if not itp:
                        if C.ndim == 3:
                            C = np.vstack((C, [sg]))
                            C_subs.append(sg_subs)
                        else:
                            C = np.array((C, sg))
                            C_subs = C_subs.append(sg_subs)
                        if L.ndim == 3:
                            L = np.vstack(
                                (
                                    L,
                                    np.array(
                                        [affine_matrix_op(sg, t) for t in L1]
                                    ),
                                )
                            )
                            tmp = [sg_subs + t_subs for t_subs in L1_subs]
                            L_subs = L_subs + tmp

                        else:
                            L = np.array(
                                L,
                                np.array(
                                    [affine_matrix_op(sg, t) for t in L1]
                                ),
                            )
                            tmp = [sg_subs + t_subs for t_subs in L1_subs]
                            L_subs = L_subs + tmp
                        more = True
    return L, L_subs
