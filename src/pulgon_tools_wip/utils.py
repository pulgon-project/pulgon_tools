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
import time

import ase
import cvxpy as cp
import numpy as np
import scipy.interpolate
import scipy.sparse as ss
import sympy
from ase import Atoms
from ipdb import set_trace
from phonopy.units import VaspToTHz
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import find_in_coord_list
from scipy.linalg.interpolative import svd
from sympy import symbols
from tqdm import tqdm

# from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.Irreps_tables import *


def e() -> np.ndarray:
    """
    Returns: identity matrix
    """
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return mat


def Cn(n: int | float) -> np.ndarray:
    """
    Args:
        n: rotate 2*pi/n

    Returns: rotation matrix
    """
    mat = np.array(
        [
            [np.cos(2 * np.pi / n), -np.sin(2 * np.pi / n), 0],
            [np.sin(2 * np.pi / n), np.cos(2 * np.pi / n), 0],
            [0, 0, 1],
        ]
    )
    return mat


def sigmaV() -> np.ndarray:
    """

    Returns: mirror symmetric matrix

    """
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return mat


def sigmaH() -> np.ndarray:
    """

    Returns: mirror symmetric matrix about x-y plane

    """
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    return mat


def U() -> np.ndarray:
    """

    Returns: A symmetric matrix about the x-axis

    """
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return mat


def U_d(fid: float | int) -> np.ndarray:
    """

    Args:
        fid: the angle between symmetry axis d and axis x, d located in th x-y plane

    Returns: A symmetric matrix about the d-axis

    """
    mat = np.array(
        [
            [np.cos(2 * fid), np.sin(2 * fid), 0],
            [np.sin(2 * fid), -np.cos(2 * fid), 0],
            [0, 0, -1],
        ]
    )
    return mat


def S2n(n: int | float) -> np.ndarray:
    """
    Args:
        n: dihedral group, rotate 2*pi/n

    Returns: rotation and mirror matrix

    """
    mat = np.array(
        [
            [np.cos(np.pi / n), np.sin(np.pi / n), 0],
            [-np.sin(np.pi / n), np.cos(np.pi / n), 0],
            [0, 0, -1],
        ]
    )
    return mat


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


def get_center_of_mass_periodic(atom):
    L = np.array([1, 1, 1])
    x = atom.get_scaled_positions()
    theta = 2.0 * np.pi * x / L
    mass = atom.get_masses()
    mtheta = (
        np.arctan2(
            (-np.sin(theta) * np.expand_dims(mass, axis=1)).sum(axis=0)
            / len(theta),
            (-np.cos(theta) * np.expand_dims(mass, axis=1)).sum(axis=0)
            / len(theta),
        )
        + np.pi
    )
    center = L * mtheta / 2.0 / np.pi

    tmp = atom.get_center_of_mass(scaled=True)
    center[2] = tmp[2]
    return center


def find_axis_center_of_nanotube(atom: ase.atoms.Atoms) -> ase.atoms.Atoms:
    """remove the center of structure to (x,y):(0.5,0.5)
    Args:
        atom: initial structure

    Returns: centralized structure

    """
    n_st = atom.copy()
    center = get_center_of_mass_periodic(atom)
    pos = (
        # np.remainder(atom.get_scaled_positions() - center + 0.5, [1, 1, 1])
        np.remainder(
            atom.get_scaled_positions()
            # - [center[0],center[1],0] + [0.5, 0.5, 0],
            - center + [0.5, 0.5, 0.5],
            [1, 1, 1],
        )
        @ atom.cell
    )

    atoms = Atoms(
        cell=n_st.cell,
        numbers=n_st.numbers,
        positions=pos,
    )
    return atoms


def atom_move_z(atom):
    n_st = atom.copy()

    pos = (
        np.remainder(
            atom.get_scaled_positions()
            - [0, 0, atom.get_scaled_positions()[0][2]],
            [1, 1, 1],
        )
        @ atom.cell
    )

    atoms = Atoms(
        cell=n_st.cell,
        numbers=n_st.numbers,
        positions=pos,
    )
    return atoms


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


def get_perms_from_ops(atoms, ops_sym, symprec=1e-2):
    """get the permutation table from symmetry operations

    Args:
        atoms:
        symprec:

    Returns: permutation table
    """
    natoms = len(atoms.numbers)
    coords_scaled = atoms.get_scaled_positions()
    coords_scaled_center = np.remainder(
        coords_scaled - [0.5, 0.5, 0], [1, 1, 1]
    )

    perms = []
    for ii, op in enumerate(ops_sym):
        tmp_perm = np.zeros((1, len(atoms.numbers)))[0]
        for jj, site in enumerate(atoms):
            pos = (site.scaled_position - [0.5, 0.5, 0]) @ atoms.cell

            tmp = op.operate(pos)
            tmp1 = np.remainder(tmp @ np.linalg.inv(atoms.cell), [1, 1, 1])
            idx2 = find_in_coord_list(coords_scaled_center, tmp1, symprec)

            if idx2.size == 0:
                idx = coords_scaled_center[:, 2] == tmp1[2]
                res = coords_scaled_center[idx]
                set_trace()
                logging.ERROR("tolerance exceed while calculate perms")
            tmp_perm[jj] = idx2

        idx = len(np.unique(tmp_perm))
        if idx != natoms:
            logging.ERROR("perms numebr != natoms")
        perms.append(tmp_perm)
    perms_table = np.array(perms).astype(np.int32)
    return perms_table


def get_matrices(atoms, ops_sym):
    perms_table = get_perms_from_ops(atoms, ops_sym)
    natoms = len(atoms.numbers)
    matrices = []
    for ii, perm in enumerate(perms_table):
        matrix = np.zeros((3 * natoms, 3 * natoms))
        for jj in range(natoms):
            idx = perm[jj]
            matrix[3 * idx : 3 * (idx + 1), 3 * jj : 3 * (jj + 1)] = ops_sym[
                ii
            ].rotation_matrix.copy()
            # matrix[4 * idx : 4 * (idx + 1), 4 * jj : 4 * (jj + 1)] = ops_sym[ii].affine_matrix.copy()
        matrices.append(matrix)
    return matrices


def affine_matrix_op(af1, af2):
    """Definition of group multiplication

    Args:
        af1:
        af2:

    Returns:

    """
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

                    itp = (abs((sg - L).sum(axis=1).sum(axis=1)) < 0.001).any()
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


def brute_force_generate_group(generators: np.ndarray, symec: float = 0.01):
    e_in = np.eye(4)
    G = generators
    L = np.array([e_in])
    while True:
        numL_old = len(L)
        for g in L:
            for h in G:
                gh = affine_matrix_op(g, h)
                judge = (abs(L - gh) < symec).all(axis=1).all(axis=1).any()
                if not judge:
                    L = np.concatenate((L, [gh]), axis=0)
        numL_new = len(L)
        if numL_old == numL_new:
            break
    return L


def brute_force_generate_group_subsquent(
    generators: np.ndarray, symec: float = 0.01
):
    e_in = np.eye(4)
    G = generators
    L = np.array([e_in])
    L_seq = [[0]]
    while True:
        numL_old = len(L)
        for ii, g in enumerate(L):
            tmp_seq1 = L_seq[ii]
            for jj, h in enumerate(G):
                gh = affine_matrix_op(g, h)
                tmp_seq2 = tmp_seq1 + [jj + 1]
                judge = (abs(L - gh) < symec).all(axis=1).all(axis=1).any()
                if not judge:
                    L = np.concatenate((L, [gh]), axis=0)
                    L_seq.append(tmp_seq2)
        numL_new = len(L)
        if numL_old == numL_new:
            break
    return L, L_seq


def dimino_affine_matrix(
    generators: np.ndarray, symec: float = 0.01
) -> np.ndarray:
    """

    Args:
        generators: the generators of point group
        symec: system precision

    Returns: all the group elements and correspond character

    """
    e_in = np.eye(4)

    G = generators
    g, g1 = G[0].copy(), G[0].copy()
    L = np.array([e_in])
    while not ((g - e_in) < symec).all():
        L = np.vstack((L, [g]))
        g = affine_matrix_op(g, g1)
    for ii in range(len(G)):
        C = np.array([e_in])
        L1 = L.copy()
        more = True
        while more:
            more = False
            for g in list(C):
                for ss in G[: ii + 1]:
                    sg = affine_matrix_op(ss, g)
                    itp = (abs((sg - L).sum(axis=1).sum(axis=1)) < symec).any()
                    if not itp:
                        if C.ndim == 3:
                            C = np.vstack((C, [sg]))
                        if L.ndim == 3:
                            L = np.vstack(
                                (
                                    L,
                                    np.array(
                                        [affine_matrix_op(sg, t) for t in L1]
                                    ),
                                )
                            )
                        more = True
    L = np.unique(L, axis=0)
    return L


def dimino_affine_matrix_and_subsquent(
    generators: np.ndarray, symec: float = 0.001
) -> np.ndarray:
    """

    Args:
        generators: the generators of point group
        symec: system precision

    Returns: all the group elements and correspond character

    """
    e_in = np.eye(4)

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

                    itp = (abs((sg - L).sum(axis=1).sum(axis=1)) < 0.001).any()
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


def get_character(qpoints, nrot, order, family, a):
    characters, paras_values, paras_symbols = line_group_sympy(
        family, qpoints, nrot, a, order
    )
    return characters, paras_values, paras_symbols


def fast_orth(A, maxrank, num):
    """Reimplementation of scipy.linalg.orth() which takes only the vectors with
    values almost equal to the maximum, and returns at most maxrank vectors.
    """
    u, s, vh = svd(A, maxrank)
    reference = s[0]
    return u[:, :num]  # Todo: correct the number
    # for i in range(s.size):
    #     if abs(reference - s[i]) > 0.8 * reference:
    #         return u[:, :i]
    # return u


def get_sym_constrains_matrices_M(ops, permutations, diminsion=3):
    """M K = 0

    :param ops:
    :param permutations:
    :param diminsion:
    :return:

    """

    if permutations.ndim == 2:
        natom = len(permutations[0])
    elif permutations.ndim == 1:
        natom = len(permutations)
        permutations = np.array([permutations])
    else:
        logging.ERROR("error for permutations' ndim")

    if len(ops.shape) == 2:
        ops = np.array([ops])

    size1 = diminsion**2
    I = np.eye(size1)
    M = []

    idx1 = np.repeat(np.arange(natom), natom)
    idx2 = np.tile(np.arange(natom), natom)

    tmp1 = (idx1 * natom + idx2) * size1
    tmp2 = (idx1 * natom + idx2 + 1) * size1
    tmp3 = np.linspace(tmp1, tmp2, size1 + 1).astype(np.int64)[:-1, :].T
    # tmp3 = np.array([np.arange(tmp1[ii], tmp2[ii]) for ii in range(len(tmp1))])
    for ii, op in enumerate(ops):
        print("now run in %s operarion" % ii)
        perm = permutations[ii]
        C = np.einsum(
            "ij,kl->ikjl",
            op[:diminsion, :diminsion],
            op[:diminsion, :diminsion],
        ).reshape(size1, size1)
        x = ss.csc_matrix((size1 * natom**2, size1 * natom**2))
        if (perm == np.arange(natom)).all():
            # x[np.arange(size1*(natom**2)), np.arange(size1*(natom**2))] = 1
            M.append(x)
            continue
        # for ii, jj in list(itertools.product(np.arange(natom), np.arange(natom))):
        #     x[(ii * natom + jj) * size1:(ii * natom + jj + 1) * size1, (ii * natom + jj) * size1:(ii * natom + jj + 1) * size1] = C
        #     pii, pjj = perm[ii], perm[jj]
        #     x[(ii * natom + jj) * size1:(ii * natom + jj + 1) * size1, (pii * natom + pjj) * size1:(pii * natom + pjj + 1) * size1] = -I
        # ptmp1 = np.hstack(([(pidx1[ii] * natom + pidx2) * size1 for ii in range(len(pidx1))]))
        # ptmp2 = np.hstack(([(pidx1[ii] * natom + pidx2 + 1) * size1 for ii in range(len(pidx1))]))

        pidx1 = perm[idx1]
        pidx2 = perm[idx2]

        ptmp1 = (pidx1 * natom + pidx2) * size1
        ptmp2 = (pidx1 * natom + pidx2 + 1) * size1

        # ptmp3 = np.array([np.arange(ptmp1[ii], ptmp2[ii]) for ii in range(len(ptmp1))])
        ptmp3 = np.linspace(ptmp1, ptmp2, size1 + 1).astype(np.int64)[:-1, :].T

        itp1 = np.repeat(tmp3, size1, axis=1)
        itp2 = np.tile(tmp3, (1, size1))
        pitp2 = np.tile(ptmp3, (1, size1))
        xl = x.tolil()
        xl[itp1, itp2] = C.flatten()
        xl[itp1, pitp2] = -I.flatten()  #
        M.append(xl)

    M = scipy.sparse.vstack((M))
    return M


def get_sym_constrains_matrices_M_for_conpact_fc(
    IFC, ops_sym, perms_ops, perms_trans, p2s_map, natom_pri, diminsion=3
):
    """

    :param ops:
    :param perms_ops:
    :param perms_trans:
    :param p2s_map:
    :param natom_pri:
    :param diminsion:
    :return:

    """

    natom = perms_ops.shape[1]
    supercell = perms_trans.shape[0]
    size1 = diminsion**2
    I = np.eye(size1)
    M = []
    res = 0

    idx1 = np.repeat(np.arange(natom_pri), natom)
    idx2 = np.tile(np.arange(natom), natom_pri)

    tmp1 = (idx1 * natom + idx2) * size1
    tmp2 = (idx1 * natom + idx2 + 1) * size1
    tmp3 = np.linspace(tmp1, tmp2, size1 + 1).astype(np.int64)[:-1, :].T

    itp1 = np.repeat(tmp3, size1, axis=1)
    itp2 = np.tile(tmp3, (1, size1))
    res = 0

    # ops = np.delete(ops,[18,36,54], axis=0)
    # perms_ops = np.delete(perms_ops,[18,36,54], axis=0)
    for ii, op in enumerate(ops_sym):
        print("now run in %s operarion" % ii)
        perm = perms_ops[ii]
        C = np.einsum(
            "ij,kl->ikjl",
            op.rotation_matrix,
            op.rotation_matrix,
        ).reshape(size1, size1)
        x = ss.csc_matrix(
            (size1 * natom * natom_pri, size1 * natom * natom_pri)
        )

        if (perm == np.arange(natom)).all():
            # M.append(x.tolil())
            continue
        pidx1 = perms_trans[:, perm[idx1 * supercell]]
        pindex1 = np.isin(pidx1, p2s_map)

        pidx1 = (pidx1.T[pindex1.T] / supercell).astype(
            np.int32
        )  # map the index i
        pidx2 = perms_trans[:, perm[idx2]].T[pindex1.T]  # map the index j

        ptmp1 = (pidx1 * natom + pidx2) * size1
        ptmp2 = (pidx1 * natom + pidx2 + 1) * size1

        # ptmp3 = np.array([np.arange(ptmp1[ii], ptmp2[ii]) for ii in range(len(ptmp1))])
        ptmp3 = np.linspace(ptmp1, ptmp2, size1 + 1).astype(np.int64)[:-1, :].T
        pitp2 = np.tile(ptmp3, (1, size1))

        xl = x.tolil()
        xl[itp1, itp2] = C.flatten()
        xl[itp1, pitp2] -= I.flatten()

        res = abs(xl.dot(IFC.flatten())).sum()
        print(res)

        if abs(res) > 1e-5:
            tmp = abs(xl.dot(IFC.flatten()))
            tmp1 = np.unique(tmp)[::-1]
            print("max value equation=%s" % max(tmp))
            M.append(xl.tocsc())

    M = scipy.sparse.vstack((M))
    M = M.tocsc()
    return M


def get_IFCSYM_from_cvxpy_M(M, IFC):
    flat_IFCs = IFC.ravel()
    x = cp.Variable(IFC.size)
    cost = cp.sum_squares(x - flat_IFCs)
    prob = cp.Problem(cp.Minimize(cost), [M @ x == 0])
    prob.solve()

    IFC_sym = x.value.reshape(IFC.shape)
    return IFC_sym


def get_freq_and_dis_from_phonopy(phonon, qpoints):
    frequencies = []
    distances = []
    for ii, q in enumerate(qpoints[0]):
        D = phonon.get_dynamical_matrix_at_q(q)
        eigvals, eigvecs = np.linalg.eigh(D)
        eigvals = eigvals.real
        frequencies.append(
            np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz
        )
        if ii == 0:
            distances.append(0)
            q_last = q.copy()
        else:
            distances.append(
                np.linalg.norm(np.dot(q - q_last, phonon.supercell.get_cell()))
            )
    frequencies = np.array(frequencies).T
    return frequencies, distances
