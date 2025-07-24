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
from typing import Union

import ase
import cvxpy as cp
import scipy as sp
import scipy.interpolate
import scipy.sparse as ss
import scipy.spatial.distance
from ase import Atoms
from ipdb import set_trace

# from phonopy.units import VaspToTHz
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import find_in_coord_list
from sympy.physics.quantum import TensorProduct

from pulgon_tools_wip.Irreps_tables import *
from pulgon_tools_wip.Irreps_tables_withparities import (
    line_group_sympy_withparities,
)


def e() -> np.ndarray:
    """
    Returns: identity matrix
    """
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return mat


def Cn(n: Union[int, float]) -> np.ndarray:
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


def U_d(fid: Union[float, int]) -> np.ndarray:
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


def S2n(n: Union[int, float]) -> np.ndarray:
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


# def get_symcell(monomer: Atoms) -> Atoms:
#     """based on the point group symmetry of monomer, return the symcell
#
#     Args:
#         monomer:
#
#     Returns: symcell
#
#     """
#     apg = LineGroupAnalyzer(monomer)
#     equ = list(apg.get_equivalent_atoms()["eq_sets"].keys())
#     # sym = apg.get_symmetry_operations()
#     return monomer[equ]


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
            - center + [0.5, 0.5, 0],
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


def get_perms_from_ops(atoms: Atoms, ops_sym, symprec=1e-2, round=4):
    """get the permutation table from symmetry operations

    Args:
        atoms:
        symprec:

    Returns: permutation table
    """
    invcell = np.linalg.inv(atoms.cell)
    coords_center = atoms.positions - atoms.get_center_of_mass()
    coords_scaled_center = coords_center @ invcell
    coords_scaled_center[coords_center @ invcell >= 0.5] -= 1
    coords_scaled_center[coords_center @ invcell <= -0.5] += 1
    coords_center = coords_scaled_center @ atoms.cell

    perms = []
    for ii, op in enumerate(ops_sym):
        perms.append([])
        operated = op.operate_multi(coords_center)
        operated_scaled = operated @ invcell
        # to me this seems much safer than the use of remainder - no floating point issues
        operated_scaled[operated @ invcell >= 0.5] -= 1
        operated_scaled[operated @ invcell <= -0.5] += 1
        operated = operated_scaled @ atoms.cell
        for aid in range(len(coords_center)):
            equivalents = np.where(
                np.linalg.norm(operated[aid] - coords_center, axis=1) < symprec
            )[0]
            if len(equivalents) == 1:
                for eq in equivalents:
                    perms[-1].append(eq)
            else:
                print(ii, aid)
                set_trace()
                raise ValueError
    perms_table = np.array(perms).astype(np.int32)
    return perms_table


def get_matrices(atoms, ops_sym, symprec=1e-5):
    perms_table = get_perms_from_ops(atoms, ops_sym, symprec=symprec)

    natoms = len(atoms.numbers)
    matrices = []
    for ii, perm in enumerate(perms_table):
        matrix = np.zeros((3 * natoms, 3 * natoms))
        for jj in range(natoms):
            idx = perm[jj]
            matrix[3 * idx : 3 * (idx + 1), 3 * jj : 3 * (jj + 1)] = ops_sym[
                ii
            ].rotation_matrix.copy()
        matrices.append(matrix)
    return matrices


def get_matrices_withPhase(atoms, ops_sym, qpoint, symprec=1e-3):
    perms_table = get_perms_from_ops(atoms, ops_sym, symprec=symprec)

    natoms = len(atoms.numbers)
    matrices = []
    for ii, perm in enumerate(perms_table):
        matrix = np.zeros((3 * natoms, 3 * natoms)).astype(np.complex128)
        phasefacter = ops_sym[ii].translation_vector[2]

        for jj in range(natoms):
            idx = perm[jj]
            matrix[3 * idx : 3 * (idx + 1), 3 * jj : 3 * (jj + 1)] = ops_sym[
                ii
            ].rotation_matrix.copy() * np.exp(1j * qpoint * phasefacter)
        matrices.append(matrix)
    return matrices


def get_modified_projector(DictParams, atom):
    family = DictParams["family"]

    if family == 4:
        nrot = DictParams["nrot"]
        g_rot = DictParams["generator_rot"]
        g_tran = DictParams["generator_tran"]

        ops_car_apg = []
        for s in range(nrot):
            for j in range(2):
                rot = np.linalg.matrix_power(
                    g_rot[0].rotation_matrix, s
                ) @ np.linalg.matrix_power(g_rot[1].rotation_matrix, j)
                tran = (
                    g_rot[0].translation_vector * s
                    + g_rot[1].translation_vector * j
                ) * atom.cell[2, 2]
                op = SymmOp.from_rotation_and_translation(
                    rotation_matrix=rot, translation_vec=tran
                )
                ops_car_apg.append(op)
        matrices_apg = get_matrices(atom, ops_car_apg)

        # res, err = [], []
        # for ii in range(len(matrices_apg)):
        #     tmp1 = np.array_equal(np.dot(D, matrices_apg[ii]), np.dot(matrices_apg[ii], D))
        #     tmp2 = np.abs(np.dot(D, matrices_apg[ii]) - np.dot(matrices_apg[ii], D)).sum()
        #
        #     res.append(tmp1)
        #     err.append(tmp2)
        # set_trace()

        ops_car_cyc = [
            SymmOp.from_rotation_and_translation(
                rotation_matrix=g_tran[0].rotation_matrix,
                translation_vec=g_tran[0].translation_vector * atom.cell[2, 2],
            )
        ]
        matrices_cyc = get_matrices(atom, ops_car_cyc)
        # m1_range = list(range(-nrot + 1, nrot + 1))
        # m1_range = list(range(1, nrot + 1))
        m1_range = list(range(-nrot + 1, 1))

        basis, dimensions = [], []
        for tmp_m1 in m1_range:
            Dmu_rot_conj, Dmu_tran_conj = get_modified_Dmu(
                DictParams, tmp_m1, symprec=1e-6
            )

            ###### generate the projector for axial point group ########
            num_modes = 0
            for ii in range(len(Dmu_rot_conj)):
                if ii == 0:
                    projector = TensorProduct(
                        Dmu_rot_conj[ii], matrices_apg[ii]
                    )
                else:
                    projector += TensorProduct(
                        Dmu_rot_conj[ii], matrices_apg[ii]
                    )

                if Dmu_rot_conj[ii].ndim != 0:
                    num_modes += (
                        Dmu_rot_conj[ii].trace() * matrices_apg[ii].trace()
                    )
                else:
                    num_modes += Dmu_rot_conj[ii] * matrices_apg[ii].trace()

            num_modes = int(num_modes.real / (2 * nrot))
            projector_apg = projector / (2 * nrot)

            ###### generate the projector for cyclic group ######
            projector_cyc = TensorProduct(Dmu_tran_conj, matrices_cyc[0])
            # projector_cyc2 = np.tensordot(Dmu_tran_conj, matrices_cyc[0], axes=0)

            # projector_cyc2 = np.einsum("ij,kl->ijkl",Dmu_tran_conj, matrices_cyc[0])
            # projector_cyc2 = projector_cyc2.transpose(0,2,1,3).reshape(projector_cyc2.shape[0]*projector_cyc2.shape[2], projector_cyc2.shape[0]*projector_cyc2.shape[2])
            # res = (projector_cyc1==projector_cyc2).all()

            # eigenvectors,eigenvalues,_ = scipy.linalg.svd(projector_cyc)
            # eigenprojectors = []
            # for i in range(len(eigenvalues)):
            #     v = eigenvectors[:, i]
            #     P = np.outer(v, v.conj())
            #     eigenprojectors.append(P)
            # eigenprojectors = np.array(eigenprojectors)
            # mask_eig = np.isclose(eigenvalues, 1, atol=1e-5)
            # projector_cyc1 =  eigenprojectors[mask_eig].sum(axis=0)

            # projector = projector_cyc1 @ projector_apg
            # projector = eigenvectors @ projector_apg
            projector = projector_cyc @ projector_apg.conj()
            # projector = projector_apg

            u, s, vh = scipy.linalg.svd(projector)
            error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(
                s[num_modes - 1]
            )

            # print("m=%s" % tmp_m1, "error=%s" % error)
            if error > 0.05:
                logging.ERROR("the error is lager than 0.05")

            if Dmu_tran_conj.ndim == 0:
                # set_trace()
                basis.append(u[:, :num_modes])
                dimensions.append(num_modes)
            else:
                tmp_basis = u[:, :num_modes]
                tmp_basis1 = np.array(np.array_split(tmp_basis, 2, axis=0))
                # tmp_basis1 = tmp_basis1[0] + tmp_basis1[1]

                # tmp_basis1 = np.array(np.array_split(tmp_basis1, 2, axis=1))[0]
                # basis_Dmu = Dmu_rot_conj[ii]
                # basis_Dmu = scipy.linalg.orth(Dmu_rot_conj[ii])
                # basis_Dmu = np.abs(basis_Dmu).reshape(-1)
                basis_Dmu = np.array([[0, 1], [1, 0]])
                basis_block1 = np.einsum(
                    "ij,jlm->ilm", basis_Dmu[0][np.newaxis], tmp_basis1
                )[0]
                # basis_partial_scalar_product = np.tensordot(basis_Dmu, basis_block, axes=([0, 1], [0, 1]))

                basis.append(basis_block1)
                dimensions.append(basis_block1.shape[1])
    elif family == 2:
        nrot = DictParams["nrot"]
        g_rot = DictParams["generator_rot"]
        g_tran = DictParams["generator_tran"]

        ops_car_apg = []
        for s in range(2 * nrot):
            rot = np.linalg.matrix_power(g_rot[0].rotation_matrix, s)
            tran = (g_rot[0].translation_vector * s) * atom.cell[2, 2]
            op = SymmOp.from_rotation_and_translation(
                rotation_matrix=rot, translation_vec=tran
            )
            ops_car_apg.append(op)
        matrices_apg = get_matrices(atom, ops_car_apg)

        ops_car_cyc = [
            SymmOp.from_rotation_and_translation(
                rotation_matrix=g_tran[0].rotation_matrix,
                translation_vec=g_tran[0].translation_vector * atom.cell[2, 2],
            )
        ]
        matrices_cyc = get_matrices(atom, ops_car_cyc)
        m1_range = list(range(int(-nrot / 2 + 1), int(nrot / 2 + 1)))

        basis, dimensions = [], []
        for tmp_m1 in m1_range:
            Dmu_rot_conj, Dmu_tran_conj = get_modified_Dmu(
                DictParams, tmp_m1, symprec=1e-6
            )
            ###### generate the projector for axial point group ########
            num_modes = 0
            tmp = []
            for ii in range(len(Dmu_rot_conj)):
                if ii == 0:
                    projector = TensorProduct(
                        matrices_apg[ii], Dmu_rot_conj[ii]
                    )
                else:
                    projector += TensorProduct(
                        matrices_apg[ii], Dmu_rot_conj[ii]
                    )

                if Dmu_rot_conj[ii].ndim != 0:
                    num_modes += (
                        Dmu_rot_conj[ii].trace() * matrices_apg[ii].trace()
                    )
                    tmp.append(matrices_apg[ii].trace())
                else:
                    num_modes += Dmu_rot_conj[ii] * matrices_apg[ii].trace()

            num_modes = int(num_modes.real / (2 * nrot))
            projector_apg = projector / (2 * nrot)

            ###### generate the projector for cyclic group ######
            projector_cyc = TensorProduct(matrices_cyc[0], Dmu_tran_conj)
            # projector_cyc2 = np.einsum("ij,kl->ijkl",Dmu_tran_conj, matrices_cyc[0])
            # projector_cyc2 = projector_cyc2.transpose(0,2,1,3).reshape(projector_cyc2.shape[0]*projector_cyc2.shape[2], projector_cyc2.shape[0]*projector_cyc2.shape[2])

            # projector = projector_cyc1 @ projector_apg
            # projector = eigenvectors @ projector_apg
            # projector = projector_cyc @ projector_apg
            projector = projector_apg

            u, s, vh = scipy.linalg.svd(projector)
            error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(
                s[num_modes - 1]
            )

            if error > 0.05:
                # set_trace()
                logging.ERROR("the error is lager than 0.05")

            if Dmu_tran_conj.ndim == 0:
                basis.append(u[:, :num_modes])
                dimensions.append(num_modes)
            else:
                tmp_basis = u[:, :num_modes]
                tmp_basis1 = np.array(np.array_split(tmp_basis, 2, axis=0))
                basis_Dmu = np.array([[0, 1], [1, 0]])

                basis_block1 = np.einsum(
                    "ij,jlm->ilm", basis_Dmu[0][np.newaxis], tmp_basis1
                )[0]
                # basis_partial_scalar_product = np.tensordot(basis_Dmu, basis_block, axes=([0, 1], [0, 1]))

                basis.append(basis_block1)
                dimensions.append(basis_block1.shape[1])

    adapted = np.concatenate(basis, axis=1)
    # if adapted.shape[0] != adapted.shape[1]:
    #     set_trace()
    return adapted, dimensions


def affine_matrix_op(af1, af2, symprec=1e-8):
    """Definition of group multiplication

    Args:
        af1: group element 1
        af2: group element 2

    Returns:

    """
    af1 = np.round(af1 / symprec).astype(int) * symprec
    af2 = np.round(af2 / symprec).astype(int) * symprec

    ro = af2[:3, :3] @ af1[:3, :3]
    # ro = af1[:3, :3] @ af2[:3, :3]
    tran = np.remainder(af2[:3, 3] + af2[:3, :3] @ af1[:3, 3], [1, 1, 1])
    # tran = (af2[:3, 3] + af2[:3, :3] @ af1[:3, 3]) % 1
    if np.isclose(tran[0], 1):
        set_trace()
    # tran = np.round(tran / symprec).astype(int) * symprec

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
    """generate all the group elements by brute force algorithm

    Args:
        generators: The most basic elements used to generate complete groups
        symec: tolerance

    Returns:
        L: all group elements
        L_seq: the multiplication order of the generators
    """
    e_in = np.eye(4)
    G = generators
    L = np.array([e_in])
    L_seq = [[0]]
    while True:
        numL_old = len(L)
        for ii, g in enumerate(L):
            tmp_seq1 = L_seq[ii]
            for jj, h in enumerate(G):
                gh = affine_matrix_op(g, h, symprec=1e-5)
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


def get_character(DictParams, symprec=1e-8):
    characters, paras_values, paras_symbols = line_group_sympy(
        DictParams, symprec
    )
    return characters, paras_values, paras_symbols


def get_character_withparities(DictParams, symprec=1e-8):
    characters, paras_values, paras_symbols = line_group_sympy_withparities(
        DictParams, symprec
    )
    return characters, paras_values, paras_symbols


def get_character_num(DictParams, symprec=1e-8):
    representation_mat, paras_values, paras_symbols = line_group_sympy(
        DictParams, symprec
    )

    characters = []
    for ii, rep_mat in enumerate(representation_mat):  # loop IR
        if rep_mat.ndim == 1:
            characters.append(rep_mat)
        else:
            characters.append(np.trace(rep_mat, axis1=1, axis2=2))
    characters = np.array(characters)
    return characters, paras_values, paras_symbols


def get_character_num_withparities(DictParams, symprec=1e-8):
    (
        representation_mat,
        paras_values,
        paras_symbols,
    ) = line_group_sympy_withparities(DictParams, symprec)
    characters = []
    for ii, rep_mat in enumerate(representation_mat):  # loop IR
        if rep_mat.ndim == 1:
            characters.append(rep_mat)
        else:
            characters.append(np.trace(rep_mat, axis1=1, axis2=2))
    characters = np.array(characters)
    return characters, paras_values, paras_symbols


def fast_orth(A, num):
    """Reimplementation of scipy.linalg.orth() which takes only the vectors with
    values almost equal to the maximum, and returns at most maxrank vectors.
    """
    # u, s, vh = scipy.linalg.interpolative.svd(A, maxrank)
    u, s, vh = scipy.linalg.svd(A)
    error = 1 - np.abs(s[num - 1] - s[num]) / np.abs(s[num - 1])
    return u[:, :num], error


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


def _one_constrains(
    x, natom_pri, natom, perm, perms_trans, p2s_map, size1, C, I, supercell
):
    idx1 = np.repeat(np.arange(natom_pri), natom)
    idx2 = np.tile(np.arange(natom), natom_pri)

    tmp1 = (idx1 * natom + idx2) * size1
    tmp2 = (idx1 * natom + idx2 + 1) * size1
    tmp3 = np.linspace(tmp1, tmp2, size1 + 1).astype(np.int64)[:-1, :].T

    itp1 = np.repeat(tmp3, size1, axis=1)
    itp2 = np.tile(tmp3, (1, size1))

    pidx1 = np.repeat(perms_trans[:, perm[p2s_map]], natom, axis=1)
    pindex1 = np.isin(pidx1, p2s_map)

    row_indices, col_indices = np.where(pindex1 == True)
    sorted_col_indices = np.argsort(col_indices)
    pidx1 = (
        pidx1[
            row_indices[sorted_col_indices],
            col_indices[sorted_col_indices],
        ]
        / supercell
    ).astype(
        np.int32
    )  # i / supercell
    pidx2 = perms_trans[:, perm[idx2]][
        row_indices[sorted_col_indices], col_indices[sorted_col_indices]
    ]  # map the index j

    ptmp1 = (pidx1 * natom + pidx2) * size1
    ptmp2 = (pidx1 * natom + pidx2 + 1) * size1
    ptmp3 = np.linspace(ptmp1, ptmp2, size1 + 1).astype(np.int64)[:-1, :].T
    pitp2 = np.tile(ptmp3, (1, size1))

    xl = x.tolil()
    xl[itp1, itp2] = C.flatten()
    xl[itp1, pitp2] -= I.flatten()

    return xl


def get_sym_constrains_matrices_M_for_conpact_fc(
    IFC, ops_sym, perms_ops, perms_trans, p2s_map, natom_pri, dimension=3
):
    """

    :param ops:
    :param perms_ops:
    :param perms_trans:
    :param p2s_map:
    :param natom_pri:
    :param dimension:
    :return:

    """

    natom = perms_ops.shape[1]
    size1 = dimension**2
    I = np.eye(size1)
    M = []

    ops_sym = np.array(ops_sym)
    perms_ops = np.array(perms_ops)
    for ii, op in enumerate(ops_sym):
        print("now run in %s operarion" % ii)
        perm = perms_ops[ii]
        size1 = dimension * dimension
        C = np.einsum(
            "ij,kl->ikjl",
            op.rotation_matrix,
            op.rotation_matrix,
        ).reshape(size1, size1)

        rows, cols, data = [], [], []
        for i in range(natom_pri):
            image_i = perms_trans[:, perm[p2s_map[i]]]
            idx = np.isin(image_i, p2s_map)
            image_i = image_i[idx].item()
            for j in range(natom):
                image_j = perms_trans[:, perm[j]][idx].item()
                tmp1 = np.ravel_multi_index(
                    (i, j, 0, 0), (natom_pri, natom, dimension, dimension)
                )
                tmp2 = np.ravel_multi_index(
                    (np.where(p2s_map == image_i)[0].item(), image_j, 0, 0),
                    (natom_pri, natom, dimension, dimension),
                )

                rows.extend(np.repeat(np.arange(tmp1, tmp1 + size1), size1))
                cols.extend(np.tile(np.arange(tmp1, tmp1 + size1), size1))
                data.extend(C.flatten())

                rows.extend(np.repeat(np.arange(tmp1, tmp1 + size1), size1))
                cols.extend(np.tile(np.arange(tmp2, tmp2 + size1), size1))
                data.extend(-I.flatten())

        xl = ss.coo_array((data, (rows, cols)), shape=(IFC.size, IFC.size))
        xl = xl.tocsc()

        res = abs(xl.dot(IFC.flatten())).sum()
        print(res)
        # if res < 1e-8:
        # set_trace()

        tmp = abs(xl.dot(IFC.flatten()))
        print("max value equation=%s" % max(tmp))
        M.append(xl)

    M = ss.vstack((M))
    return M


def _calc_dists(atoms, tolerance=1e-3):
    """
    Return the distances between atoms in the supercell, their
    degeneracies and the associated displacements along OZ.
    """
    MIN_DELTA = -1
    MAX_DELTA = 1
    positions = atoms.positions
    cell = atoms.cell
    n_satoms = positions.shape[0]
    d2s = np.empty((MAX_DELTA - MIN_DELTA + 1, n_satoms, n_satoms))

    # TODO: This could not be enough and eventually we should do a proper check
    # for the shortest distance.
    for j, j_c in enumerate(range(MIN_DELTA, MAX_DELTA + 1)):
        shifted_positions = positions + (j_c * cell[2, :])[np.newaxis, :]
        d2s[j, :, :] = sp.spatial.distance.cdist(
            positions, shifted_positions, "sqeuclidean"
        )
        # d2s[j, :, :] = sp.spatial.distance.cdist(positions, shifted_positions)
    d2min = d2s.min(axis=0)
    dmin = np.sqrt(d2min)
    degenerate = np.abs(d2s - d2min) < tolerance
    nequi = degenerate.sum(axis=0, dtype=int)

    maxequi = nequi.max()
    shifts = np.empty((n_satoms, n_satoms, maxequi))
    sorting = np.argsort(np.logical_not(degenerate), axis=0)
    shifts = np.transpose(sorting[:maxequi, :, :], (1, 2, 0)).astype(np.intc)
    shifts = np.asarray(range(MIN_DELTA, MAX_DELTA + 1))[shifts]
    return (dmin, nequi, shifts)


def get_continum_constrains_matrices_M_for_conpact_fc(phonon):
    IFC = phonon.force_constants.copy()
    scell = phonon.supercell

    symbols = scell.symbols
    cell = scell.cell
    positions = (
        (scell.scaled_positions + np.asarray([0.5, 0.5, 0.0])[np.newaxis, :])
        % 1.0
    ) @ cell
    ase_atoms = ase.Atoms(symbols, positions, cell=cell, pbc=True)
    dists, degeneracy, shifts = _calc_dists(ase_atoms)
    n_satoms = len(symbols)

    # %%
    average_delta = np.zeros((n_satoms, n_satoms, 3))
    for i in range(n_satoms):
        for j in range(n_satoms):
            n_elements = degeneracy[i, j]
            # n_elements=1
            for i_d in range(n_elements):
                average_delta[i, j, :] += (
                    positions[j, :]
                    - positions[i, :]
                    + shifts[i, j, i_d] * cell[2, :]
                )
            average_delta[i, j, :] /= n_elements

    average_pos = scell.scaled_positions @ cell

    # %%
    average_products = np.zeros((n_satoms, n_satoms, 3, 3))
    for i in range(n_satoms):
        for j in range(n_satoms):
            n_elements = degeneracy[i, j]
            # n_elements=1
            for i_d in range(n_elements):
                delta = (
                    positions[j, :]
                    - positions[i, :]
                    + shifts[i, j, i_d] * cell[2, :]
                )
                average_products[i, j, :, :] += np.outer(delta, delta)
            average_products[i, j, :, :] /= n_elements

    M1 = ss.coo_array(([]))

    # Append the acoustic sum rules originating from translations.
    n_atoms, n_satoms = IFC.shape[:2]
    n_rows = M1.shape[0]
    rows = M1.row.tolist()
    cols = M1.col.tolist()
    data = M1.data.tolist()

    # translational sum rules
    for i in range(n_atoms):
        for alpha in range(3):
            for beta in range(3):
                for j in range(n_satoms):
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index((i, j, alpha, beta), IFC.shape)
                    )
                    data.append(1.0)
                n_rows += 1

    # The same but for rotations (Born-Huang).
    positions = phonon.supercell.positions
    for i in range(n_atoms):
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    for j in range(n_satoms):
                        r_ij = average_delta[phonon.primitive.p2s_map[i], j]
                        rows.append(n_rows)
                        cols.append(
                            np.ravel_multi_index(
                                (i, j, alpha, beta), IFC.shape
                            )
                        )
                        data.append(r_ij[gamma])
                        rows.append(n_rows)
                        cols.append(
                            np.ravel_multi_index(
                                (i, j, alpha, gamma), IFC.shape
                            )
                        )
                        data.append(-r_ij[beta])
                    n_rows += 1

    # And fthe Huang invariances, also for rotation.
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                for delta in range(3):
                    for i in range(n_atoms):
                        for j in range(n_satoms):
                            products = average_products[
                                phonon.primitive.p2s_map[i], j
                            ]
                            rows.append(n_rows)
                            cols.append(
                                np.ravel_multi_index(
                                    (i, j, alpha, beta), IFC.shape
                                )
                            )
                            data.append(products[gamma, delta])
                            rows.append(n_rows)
                            cols.append(
                                np.ravel_multi_index(
                                    (i, j, gamma, delta), IFC.shape
                                )
                            )
                            data.append(-products[alpha, beta])
                    n_rows += 1

    # Make sure the IFC matrix is symmetric.
    for i in range(n_atoms):
        for j in range(n_atoms):
            for alpha in range(3):
                for beta in range(3):
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index(
                            (i, phonon.primitive.p2s_map[j], alpha, beta),
                            IFC.shape,
                        )
                    )

                    data.append(1.0)
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index(
                            (j, phonon.primitive.p2s_map[i], beta, alpha),
                            IFC.shape,
                        )
                    )
                    data.append(-1.0)
                    n_rows += 1

    # Rebuild the sparse matrix.
    M1 = ss.coo_array((data, (rows, cols)), shape=(n_rows, IFC.size))
    return M1


def get_IFCSYM_from_cvxpy_M(M, IFC):
    flat_IFCs = IFC.ravel()
    x = cp.Variable(IFC.size)
    cost = cp.sum_squares(x - flat_IFCs)
    prob = cp.Problem(cp.Minimize(cost), [M @ x == 0])
    prob.solve()
    IFC_sym = x.value.reshape(IFC.shape)
    return IFC_sym


# def get_freq_and_dis_from_phonopy(phonon, qpoints):
#     frequencies = []
#     distances = []
#     for ii, q in enumerate(qpoints[0]):
#         D = phonon.get_dynamical_matrix_at_q(q)
#         eigvals, eigvecs = np.linalg.eigh(D)
#         eigvals = eigvals.real
#         frequencies.append(
#             np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz
#         )
#         if ii == 0:
#             distances.append(0)
#             q_last = q.copy()
#         else:
#             distances.append(
#                 np.linalg.norm(np.dot(q - q_last, phonon.supercell.get_cell()))
#             )
#     frequencies = np.array(frequencies).T
#     return frequencies, distances


def get_independent_atoms(perms):
    atom_num = []
    if perms.ndim == 2:
        for line in perms.T:
            atom_num.append(np.unique(line)[0])
        return np.unique(atom_num)
    else:
        return perms


def get_site_symmetry(atom_num, perms_ops, ops_sym):
    site_symmetry = []
    for num in atom_num:
        idx = np.where(perms_ops[:, num] == num)[0]
        site_symmetry.append(
            np.array([ops_sym[ii].rotation_matrix for ii in idx])
        )
    return site_symmetry


def get_symbols_from_ops(ops_sym):
    """
    Get the symbols of point group operations from the rotation matrix.

    Args:
        ops_sym: A list of rotation matrix of symmetry operations.

    Returns:
        symbols: A list of point group symbols.

    Notes:
        The point group operations are as follows:
        E: Identity
        sigmaH: Reflection in the horizontal mirror plane.
        sigmaV: Reflection in the vertical mirror plane.
        U: Reflection in the plane perpendicular to the axis.
        Cn: Rotation by 2 * pi / n about the axis.
        Sn: Rotation by 2 * pi / n about the axis followed by reflection in the plane perpendicular to the axis.
    """
    symbols = []
    for op in ops_sym:
        op_ro = op[:3, :3]

        val1, _ = np.linalg.eig(op_ro)
        itp1 = np.isclose(val1, 1, atol=1e-4)
        itp2 = np.isclose(val1, -1, atol=1e-4)

        # set_trace()
        if np.logical_or(itp1, itp2).all():  # test if it is a reflection h,v,U
            if itp1.sum() == 3:
                symbols.append("E")
            elif itp1.sum() == 2 and itp2.sum() == 1:
                tmp = np.where(itp2)[0].item()
                if tmp == 2:
                    symbols.append("sigmaH")
                else:
                    symbols.append("sigmaV")
            elif itp1.sum() == 1 and itp2.sum() == 2 and itp2[2]:
                symbols.append("U")
            else:
                logging.ERROR("The Eigvenvalues are very strange")
        else:  # test if it is a rotation Cn
            tmp = 1 / (-np.log(val1) / 2 / np.pi * 1j)[:2]

            if np.isclose(
                tmp[0].real, abs(tmp[1].real), atol=1e-4
            ) and np.isclose(val1[2].real, 1, atol=1e-4):
                num = int(np.round(tmp[0].real))
                symbols.append("C%d" % num)
            elif np.isclose(
                tmp[0].real, abs(tmp[1].real), atol=1e-4
            ) and np.isclose(val1[2].real, -1, atol=1e-4):
                num = int(np.round(tmp[0].real))
                symbols.append("S%d" % num)
            else:
                logging.ERROR("The Eigvenvalues are very strange")
    return symbols


def divide_irreps(vec, adapted, dimensions):
    """
    Project vectors into IR space according to the adapted basis.

    Parameters
    ----------
    vec : array of shape (n,m), where n is the number of vectors and m is the dimension
        The vector to be divided
    adapted : array of shape (m,m). The basis vectors are arranged in columns.
        The adapted basis
    dimensions : list of int. Sum(dimensions) = m.
        The dimension of each irrep

    Returns
    -------
    means : array of shape (n,k), where k is the number of irreps
        The projected length of each irrep
    """
    tmp1 = vec @ adapted.conj()
    # tmp1 = vec @ adapted
    start = 0
    means, vectors = [], []
    for im, dim in enumerate(dimensions):
        end = start + dim
        if vec.ndim == 1:
            means.append((np.abs(tmp1[start:end]) ** 2).sum())
            # vectors.append((tmp1 * adapted)[:, start:end].sum(axis=1))
        else:
            means.append((np.abs(tmp1[:, start:end]) ** 2).sum(axis=1))
        start = copy.copy(end)
    means = np.array(means)
    if means.ndim > 1:
        means = means.T
    return np.array(means)


def get_p_from_qrn(q, r, n):
    q_tilder = q / n  # q_tilder = a / f
    if np.isclose(q_tilder, int(q_tilder)):
        q_tilder = int(q_tilder)
    else:
        logging.ERROR("q_tilder is not an interger")
    p_tilder = r ** (sympy.totient(q_tilder) - 1)
    p = n * p_tilder
    return p


def angle_between_points(A, B, C):
    """
    Calculate angle between three points. Point b is the vertex

    Parameters
    ----------
    A : The first point coordinates
    B : The second point coordinates
    C : The third point coordinates

    Returns
    -------
    angle : float
        The angle between points A, B and C in degrees.
    """
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)
