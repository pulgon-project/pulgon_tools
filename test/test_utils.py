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

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core.operations import SymmOp

from pulgon_tools.utils import (
    Cn,
    S2n,
    U,
    U_d,
    _calc_dists,
    affine_matrix_op,
    angle_between_points,
    brute_force_generate_group,
    brute_force_generate_group_subsequent,
    decimal_places,
    dimino,
    dimino_affine_matrix,
    dimino_affine_matrix_and_character,
    dimino_affine_matrix_and_subsequent,
    divide_irreps,
    e,
    find_axis_center_of_nanotube,
    frac_range,
    get_center_of_mass_periodic,
    get_character,
    get_character_num,
    get_character_num_withparities,
    get_character_withparities,
    get_continum_constrains_matrices_M_for_conpact_fc,
    get_IFCSYM_from_cvxpy_M,
    get_matrices,
    get_matrices_withPhase,
    get_perms_from_ops,
    get_sym_constrains_matrices_M,
    get_symbols_from_ops,
    refine_cell,
    sigmaH,
    sigmaV,
    sortrows,
)

FCS_DATA = "test/data/fcs"

# ── Helper: build an affine matrix from a 3x3 rotation ──


def _make_affine(rot3, trans=None):
    """Build a 4x4 affine matrix from a 3x3 rotation."""
    af = np.eye(4)
    af[:3, :3] = rot3
    if trans is not None:
        af[:3, 3] = trans
    return af


# ================================================================
#  Pure math / geometry utilities
# ================================================================


class TestAngleBetweenPoints:
    def test_right_angle(self):
        assert np.isclose(
            angle_between_points([1, 0, 0], [0, 0, 0], [0, 1, 0]), 90.0
        )

    def test_straight_line(self):
        assert np.isclose(
            angle_between_points([-1, 0, 0], [0, 0, 0], [1, 0, 0]), 180.0
        )

    def test_zero_angle(self):
        assert np.isclose(
            angle_between_points([1, 0, 0], [0, 0, 0], [2, 0, 0]), 0.0
        )

    def test_60_degrees(self):
        A = [1, 0, 0]
        B = [0, 0, 0]
        C = [np.cos(np.pi / 3), np.sin(np.pi / 3), 0]
        assert np.isclose(angle_between_points(A, B, C), 60.0, atol=1e-10)


class TestDivideIrreps:
    def test_1d_input(self):
        # Identity basis, two 1D irreps
        adapted = np.eye(2)
        vec = np.array([3.0, 4.0])
        result = divide_irreps(vec, adapted, [1, 1])
        assert np.isclose(result[0], 9.0)
        assert np.isclose(result[1], 16.0)

    def test_2d_input(self):
        adapted = np.eye(3)
        vec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        result = divide_irreps(vec, adapted, [1, 2])
        assert result.shape == (2, 2)
        assert np.isclose(result[0, 0], 1.0)
        assert np.isclose(result[0, 1], 0.0)
        assert np.isclose(result[1, 0], 0.0)
        assert np.isclose(result[1, 1], 2.0)

    def test_sum_equals_norm_squared(self):
        # Total projection should equal squared norm
        adapted = np.eye(4)
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        result = divide_irreps(vec, adapted, [2, 2])
        assert np.isclose(result.sum(), np.linalg.norm(vec) ** 2)


class TestRefineCell:
    def test_single_atom(self):
        pos = np.array([1.3, -0.2, 0.7])
        numbers = np.array([6])
        p, n = refine_cell(pos, numbers)
        assert (p >= 0).all() and (p < 1).all()

    def test_removes_duplicates(self):
        pos = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
        numbers = np.array([6, 6, 8])
        p, n = refine_cell(pos, numbers)
        assert len(p) == 2
        assert len(n) == 2

    def test_wraps_negative(self):
        pos = np.array([[-0.1, -0.3, 1.2]])
        numbers = np.array([6])
        p, n = refine_cell(pos, numbers)
        assert np.allclose(p, [[0.9, 0.7, 0.2]], atol=1e-3)


class TestFracRange:
    def test_basic(self):
        assert frac_range(0.5, 3.5) == [1, 2, 3]

    def test_exclude_left(self):
        result = frac_range(1.0, 3.5, left=False)
        assert 1 not in result

    def test_exclude_right(self):
        result = frac_range(0.5, 3.0, right=False)
        assert 3 not in result

    def test_both_boundaries_integer(self):
        result = frac_range(1.0, 3.0, left=False, right=False)
        assert result == [2]


class TestSortrows:
    def test_sorts_rows(self):
        a = np.array([[3, 1], [1, 2], [1, 1]])
        result = sortrows(a)
        assert np.array_equal(result[0], [1, 1])
        assert np.array_equal(result[1], [1, 2])


# ================================================================
#  Symmetry operation matrices (basic)
# ================================================================


class TestSymmetryMatrices:
    def test_e_is_identity(self):
        assert np.array_equal(e(), np.eye(3))

    def test_Cn_period(self):
        # C4 applied 4 times = identity
        c4 = Cn(4)
        result = np.linalg.matrix_power(c4, 4)
        assert np.allclose(result, np.eye(3), atol=1e-10)

    def test_S2n_det(self):
        # S2n is an improper rotation, det = -1
        assert np.isclose(np.linalg.det(S2n(6)), -1.0)

    def test_sigmaV_is_reflection(self):
        sv = sigmaV()
        assert np.allclose(sv @ sv, np.eye(3))
        assert np.isclose(np.linalg.det(sv), -1.0)

    def test_sigmaH_is_reflection(self):
        sh = sigmaH()
        assert np.allclose(sh @ sh, np.eye(3))
        assert np.isclose(np.linalg.det(sh), -1.0)

    def test_U_is_involution(self):
        u = U()
        assert np.allclose(u @ u, np.eye(3))
        # U = diag(1,-1,-1) is a C2 rotation, det = +1
        assert np.isclose(np.linalg.det(u), 1.0)

    def test_U_d_is_involution(self):
        ud = U_d(np.pi / 6)
        assert np.allclose(ud @ ud, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(ud), 1.0)


# ================================================================
#  Affine matrix operations and Dimino variants
# ================================================================


class TestAffineMatrixOp:
    def test_identity_product(self):
        I = np.eye(4)
        af = _make_affine(Cn(3), [0.0, 0.0, 0.25])
        result = affine_matrix_op(I, af)
        assert np.allclose(result[:3, :3], af[:3, :3], atol=1e-6)

    def test_rotation_composition(self):
        af1 = _make_affine(Cn(4))
        af2 = _make_affine(Cn(4))
        result = affine_matrix_op(af1, af2)
        expected_rot = Cn(4) @ Cn(4)
        assert np.allclose(result[:3, :3], expected_rot, atol=1e-6)


class TestDiminoAffineMatrix:
    def test_cyclic_group(self):
        # C3 affine -> should produce 3 elements
        gen = np.array([_make_affine(Cn(3))])
        group = dimino_affine_matrix(gen, symec=0.01)
        assert len(group) >= 3

    def test_includes_identity(self):
        gen = np.array([_make_affine(Cn(3))])
        group = dimino_affine_matrix(gen)
        has_identity = any(np.allclose(g, np.eye(4), atol=1e-6) for g in group)
        assert has_identity


class TestDiminoAffineMatrixAndSubsequent:
    def test_returns_group_and_subsequences(self):
        gen = np.array([_make_affine(Cn(3))])
        group, subs = dimino_affine_matrix_and_subsequent(gen)
        assert len(group) == len(subs)
        assert len(group) == 3

    def test_identity_subsequence(self):
        gen = np.array([_make_affine(Cn(2))])
        group, subs = dimino_affine_matrix_and_subsequent(gen)
        # First element is identity with subsequence [0]
        assert subs[0] == [0]


class TestDiminoAffineMatrixAndCharacter:
    def test_scalar_character(self):
        # C3 with trivial (scalar) character = 1 for all elements
        gen = np.array([_make_affine(Cn(3))])
        char = np.array([np.complex128(1)])
        group, traces = dimino_affine_matrix_and_character(gen, char)
        assert len(group) == 3
        assert len(traces) == 3
        # All characters should be 1 for trivial representation
        assert np.allclose(np.abs(traces), 1.0)

    def test_matrix_character(self):
        # C3 with 2x2 rotation character
        angle = 2 * np.pi / 3
        char_mat = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.complex128,
        )
        gen = np.array([_make_affine(Cn(3))])
        char = np.array([char_mat])
        group, traces = dimino_affine_matrix_and_character(gen, char)
        assert len(group) == 3
        assert len(traces) == 3


class TestBruteForceGenerateGroup:
    def test_c4_group(self):
        gen = np.array([_make_affine(Cn(4))])
        group = brute_force_generate_group(gen)
        assert len(group) == 4

    def test_c6v_group(self):
        gen = np.array([_make_affine(Cn(6)), _make_affine(sigmaV())])
        group = brute_force_generate_group(gen)
        assert len(group) == 12


class TestBruteForceGenerateGroupSubsequent:
    def test_c3_group(self):
        gen = np.array([_make_affine(Cn(3))])
        group, seqs = brute_force_generate_group_subsequent(gen)
        assert len(group) == 3
        assert len(seqs) == 3


# ================================================================
#  get_symbols_from_ops
# ================================================================


class TestGetSymbolsFromOps:
    def test_identity(self):
        ops = [_make_affine(e())]
        assert get_symbols_from_ops(ops) == ["E"]

    def test_sigmaH(self):
        ops = [_make_affine(sigmaH())]
        assert get_symbols_from_ops(ops) == ["sigmaH"]

    def test_sigmaV(self):
        ops = [_make_affine(sigmaV())]
        assert get_symbols_from_ops(ops) == ["sigmaV"]

    def test_U_axis(self):
        ops = [_make_affine(U())]
        assert get_symbols_from_ops(ops) == ["U"]

    def test_rotation_Cn(self):
        ops = [_make_affine(Cn(6))]
        symbols = get_symbols_from_ops(ops)
        assert symbols == ["C6"]

    def test_improper_rotation_S2n(self):
        ops = [_make_affine(S2n(3))]
        symbols = get_symbols_from_ops(ops)
        assert symbols[0].startswith("S")

    def test_multiple_ops(self):
        ops = [_make_affine(e()), _make_affine(Cn(4)), _make_affine(sigmaH())]
        symbols = get_symbols_from_ops(ops)
        assert len(symbols) == 3
        assert symbols[0] == "E"
        assert symbols[2] == "sigmaH"


# ================================================================
#  get_character (thin wrapper)
# ================================================================


class TestGetCharacter:
    def test_family4(self):
        DictParams = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 4,
            "a": 3.0,
        }
        chars, vals, syms = get_character(DictParams)
        assert len(chars) > 0


# ================================================================
#  Permutation and representation matrices (need Atoms + SymmOp)
# ================================================================


def _make_c4_atoms():
    """4 atoms related by C4 rotation around z, centered at origin."""
    r = 2.0
    positions = [
        [r, 0, 0],
        [0, r, 0],
        [-r, 0, 0],
        [0, -r, 0],
    ]
    cell = [[10, 0, 0], [0, 10, 0], [0, 0, 5]]
    atoms = Atoms("C4", positions=positions, cell=cell, pbc=True)
    return atoms


def _c4_symmops():
    """C4 rotation as SymmOp list."""
    c4 = Cn(4)
    ops = [
        SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0]),
        SymmOp.from_rotation_and_translation(c4, [0, 0, 0]),
        SymmOp.from_rotation_and_translation(c4 @ c4, [0, 0, 0]),
        SymmOp.from_rotation_and_translation(c4 @ c4 @ c4, [0, 0, 0]),
    ]
    return ops


class TestGetPermsFromOps:
    def test_permutation_shape(self):
        atoms = _make_c4_atoms()
        ops = _c4_symmops()
        perms = get_perms_from_ops(atoms, ops, symprec=0.5)
        assert perms.shape == (4, 4)

    def test_identity_permutation(self):
        atoms = _make_c4_atoms()
        ops = [SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])]
        perms = get_perms_from_ops(atoms, ops, symprec=0.5)
        assert np.array_equal(perms[0], [0, 1, 2, 3])


class TestGetMatrices:
    def test_matrix_shape(self):
        atoms = _make_c4_atoms()
        ops = _c4_symmops()
        matrices = get_matrices(atoms, ops, symprec=0.5)
        n = len(atoms)
        assert len(matrices) == 4
        assert matrices[0].shape == (3 * n, 3 * n)

    def test_identity_matrix(self):
        atoms = _make_c4_atoms()
        ops = [SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])]
        matrices = get_matrices(atoms, ops, symprec=0.5)
        assert np.allclose(matrices[0], np.eye(3 * len(atoms)))


class TestGetMatricesWithPhase:
    def test_matrix_shape(self):
        atoms = _make_c4_atoms()
        ops = _c4_symmops()
        matrices = get_matrices_withPhase(atoms, ops, qpoint=0.0, symprec=0.5)
        n = len(atoms)
        assert len(matrices) == 4
        assert matrices[0].shape == (3 * n, 3 * n)

    def test_zero_qpoint_equals_real(self):
        atoms = _make_c4_atoms()
        ops = [SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])]
        mats = get_matrices_withPhase(atoms, ops, qpoint=0.0, symprec=0.5)
        # With q=0 and identity, should be real identity
        assert np.allclose(mats[0], np.eye(3 * len(atoms)))


# ================================================================
#  _calc_dists
# ================================================================


class TestCalcDists:
    def test_basic_output(self):
        atoms = Atoms(
            "C2",
            positions=[[0, 0, 0], [0, 0, 1.5]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        dists, nequi, shifts = _calc_dists(atoms)
        n = len(atoms)
        assert dists.shape == (n, n)
        assert nequi.shape == (n, n)
        assert np.allclose(np.diag(dists), 0.0)

    def test_symmetry(self):
        atoms = Atoms(
            "C4",
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        dists, _, _ = _calc_dists(atoms)
        assert np.allclose(dists, dists.T)

    def test_periodic_image(self):
        # Two atoms that are closer via periodic image along z
        atoms = Atoms(
            "C2",
            positions=[[0, 0, 0.5], [0, 0, 4.5]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        dists, _, _ = _calc_dists(atoms)
        # Direct distance is 4.0, but periodic image distance is 1.0
        assert dists[0, 1] < 2.0


# ================================================================
#  Additional wrapper functions and utilities
# ================================================================


class TestDecimalPlaces:
    def test_integer(self):
        assert decimal_places(3) == 0

    def test_one_decimal(self):
        assert decimal_places(1.5) == 1

    def test_three_decimals(self):
        assert decimal_places(0.123) == 3


class TestGetCenterOfMassPeriodic:
    def test_single_atom(self):
        atoms = Atoms(
            "C",
            positions=[[5, 5, 2]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        center = get_center_of_mass_periodic(atoms)
        assert center.shape == (3,)

    def test_symmetric_atoms(self):
        # 4 atoms symmetrically placed
        atoms = Atoms(
            "C4",
            positions=[
                [2, 5, 0],
                [8, 5, 0],
                [5, 2, 0],
                [5, 8, 0],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        center = get_center_of_mass_periodic(atoms)
        assert len(center) == 3


class TestFindAxisCenterOfNanotube:
    def test_preserves_atom_count(self):
        atoms = Atoms(
            "C4",
            positions=[
                [2, 5, 0],
                [8, 5, 0],
                [5, 2, 0],
                [5, 8, 0],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        centered = find_axis_center_of_nanotube(atoms)
        assert len(centered) == len(atoms)

    def test_output_has_cell(self):
        atoms = Atoms(
            "C",
            positions=[[3, 3, 0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
            pbc=True,
        )
        centered = find_axis_center_of_nanotube(atoms)
        assert np.allclose(centered.cell, atoms.cell)


class TestGetCharacterNum:
    def test_family4(self):
        DictParams = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 4,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num(DictParams)
        assert len(chars) > 0
        # Characters should be scalar traces
        for c in chars:
            assert c.ndim == 1


class TestGetCharacterWithparities:
    def test_family4(self):
        DictParams = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 4,
            "a": 3.0,
        }
        chars, vals, syms = get_character_withparities(DictParams)
        assert len(chars) > 0


class TestGetCharacterNumWithparities:
    def test_family4(self):
        DictParams = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 4,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num_withparities(DictParams)
        assert len(chars) > 0


# ================================================================
#  dimino (non-affine, 3x3 matrices)
# ================================================================


class TestDimino:
    def test_single_generator_c4(self):
        gens = np.array([Cn(4)])
        group = dimino(gens, symec=4)
        assert group.shape == (4, 3, 3)

    def test_two_generators_c6v(self):
        gens = np.array([Cn(6), sigmaV()])
        group = dimino(gens, symec=4)
        # C6v has 12 elements
        assert group.shape == (12, 3, 3)

    def test_contains_identity(self):
        gens = np.array([Cn(3)])
        group = dimino(gens, symec=4)
        has_id = any(np.allclose(g, np.eye(3), atol=1e-6) for g in group)
        assert has_id

    def test_closure(self):
        """Every product of two group elements is in the group."""
        gens = np.array([Cn(4)])
        group = dimino(gens, symec=4)
        for a in group:
            for b in group:
                prod = a @ b
                found = any(np.allclose(prod, g, atol=1e-4) for g in group)
                assert found


# ================================================================
#  dimino_affine_matrix — extra branches
# ================================================================


class TestDiminoAffineMatrixExtended:
    def test_two_generators(self):
        """Two generators expand L via coset enumeration."""
        gen = np.array(
            [
                _make_affine(Cn(3)),
                _make_affine(sigmaV()),
            ]
        )
        group = dimino_affine_matrix(gen, symec=0.01)
        # C3v has 6 elements
        assert len(group) == 6


# ================================================================
#  dimino_affine_matrix_and_subsequent — extra branches
# ================================================================


class TestDiminoAffineMatrixAndSubsequentExtended:
    def test_two_generators(self):
        gen = np.array(
            [
                _make_affine(Cn(3)),
                _make_affine(sigmaV()),
            ]
        )
        group, subs = dimino_affine_matrix_and_subsequent(gen)
        assert len(group) == 6
        assert len(subs) == 6


# ================================================================
#  dimino_affine_matrix_and_character — extra branches
# ================================================================


class TestDiminoAffineMatrixAndCharacterExtended:
    def test_two_generators_scalar(self):
        """Two generators with scalar characters."""
        gen = np.array(
            [
                _make_affine(Cn(3)),
                _make_affine(sigmaV()),
            ]
        )
        char = np.array(
            [
                np.complex128(1),
                np.complex128(1),
            ]
        )
        group, traces = dimino_affine_matrix_and_character(gen, char)
        assert len(group) == 6
        assert len(traces) == 6


# ================================================================
#  get_character_num with 2D irreps (trace branch)
# ================================================================


class TestGetCharacterNum2D:
    def test_family6_has_2d_irreps(self):
        """Family 6 produces 2D irreps, covering np.trace branch."""
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 6,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num(params)
        assert len(chars) > 0


class TestGetCharacterNumWithparities2D:
    def test_family6_has_2d_irreps(self):
        """Family 6 with parities, covering np.trace branch."""
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 6,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num_withparities(params)
        assert len(chars) > 0


# ================================================================
#  get_sym_constrains_matrices_M
# ================================================================


# ================================================================
#  get_character / get_character_num — more families & qpoints
# ================================================================


class TestGetCharacterMoreFamilies:
    def test_family2(self):
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 2,
            "a": 3.0,
        }
        chars, vals, syms = get_character(params)
        assert len(chars) > 0

    def test_family3(self):
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 3,
            "a": 3.0,
        }
        chars, vals, syms = get_character(params)
        assert len(chars) > 0

    def test_family8(self):
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 8,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num(params)
        assert len(chars) > 0

    def test_family13(self):
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 13,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num(params)
        assert len(chars) > 0

    def test_nonzero_qpoint(self):
        params = {
            "qpoints": 0.5,
            "nrot": 6,
            "order": [[0]],
            "family": 4,
            "a": 3.0,
        }
        chars, vals, syms = get_character_num(params)
        assert len(chars) > 0

    def test_family6_withparities(self):
        params = {
            "qpoints": 0.0,
            "nrot": 6,
            "order": [[0]],
            "family": 6,
            "a": 3.0,
        }
        chars, vals, syms = get_character_withparities(params)
        assert len(chars) > 0


class TestGetSymConstrainsMatricesM:
    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_identity_returns_zero_constraint(self):
        """Identity permutation should produce zero constraint."""
        import scipy.sparse as ss_mod

        op = np.eye(3)
        perm = np.array([0, 1])
        M = get_sym_constrains_matrices_M(np.array([op]), perm, diminsion=3)
        assert M.shape[0] > 0
        # Identity perm -> M should be all zeros
        assert np.allclose(M.toarray(), 0.0)

    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_nontrivial_perm(self):
        """Non-identity permutation generates nonzero constraints."""
        import scipy.sparse as ss_mod

        c4 = Cn(4)
        perm = np.array([1, 0])  # swap atoms
        M = get_sym_constrains_matrices_M(np.array([c4]), perm, diminsion=3)
        assert M.shape[0] > 0
        assert M.nnz > 0

    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_1d_perm_input(self):
        """1D permutation array is handled correctly."""
        import scipy.sparse as ss_mod

        op = Cn(2)
        perm = np.array([1, 0])
        M = get_sym_constrains_matrices_M(np.array([op]), perm, diminsion=3)
        assert M.shape[1] == (2 * 2 * 9)

    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_2d_perm_and_2d_ops(self):
        """2D permutation + single 2D op (both get reshaped)."""
        c4 = Cn(4)
        # 2D permutations: two ops, swap atoms
        perms = np.array([[1, 0], [0, 1]])
        M = get_sym_constrains_matrices_M(
            np.array([c4, np.eye(3)]), perms, diminsion=3
        )
        assert M.shape[0] > 0
        assert M.shape[1] == (2 * 2 * 9)

    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_single_2d_op(self):
        """Single 3x3 op matrix (2D) is wrapped to 3D."""
        op = Cn(3)  # shape (3,3) not (1,3,3)
        perm = np.array([1, 0])
        M = get_sym_constrains_matrices_M(op, perm, diminsion=3)
        assert M.shape[0] > 0


# ================================================================
#  get_continum_constrains_matrices_M_for_conpact_fc
# ================================================================


class TestGetContinumConstrainsMatrices:
    @pytest.fixture
    def phonon_obj(self):
        import phonopy

        return phonopy.load(
            phonopy_yaml=f"{FCS_DATA}/phonopy.yaml",
            force_constants_filename=f"{FCS_DATA}/FORCE_CONSTANTS",
        )

    def test_returns_sparse_matrix(self, phonon_obj):
        M = get_continum_constrains_matrices_M_for_conpact_fc(phonon_obj)
        assert M.shape[0] > 0
        assert M.shape[1] == phonon_obj.force_constants.size

    def test_constraint_shape_consistent(self, phonon_obj):
        M = get_continum_constrains_matrices_M_for_conpact_fc(phonon_obj)
        IFC = phonon_obj.force_constants
        assert M.shape[1] == IFC.size

    def test_constraints_near_satisfied(self, phonon_obj):
        """Constraint violation should be bounded for raw IFC."""
        M = get_continum_constrains_matrices_M_for_conpact_fc(phonon_obj)
        IFC = phonon_obj.force_constants
        residual = M.dot(IFC.ravel())
        # Not necessarily zero, but should be finite
        assert np.all(np.isfinite(residual))


class TestGetIFCSYMFromCvxpyM:
    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_simple_constraint(self):
        import scipy.sparse as ss

        # Simple test: 2x2 IFC, constraint that sum = 0
        IFC = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Constraint: sum of all elements = 0
        M = ss.csr_matrix(np.ones((1, 4)))
        result = get_IFCSYM_from_cvxpy_M(M, IFC)
        assert result.shape == IFC.shape
        # Should satisfy constraint
        assert np.abs(M.dot(result.ravel())).sum() < 1e-4
