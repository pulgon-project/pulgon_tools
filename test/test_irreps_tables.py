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
import sympy
from ase.io.vasp import read_vasp

from pulgon_tools.generate_irreps_tables import get_linegroup_symmetry_dataset
from pulgon_tools.Irreps_tables import frac_range, line_group_sympy
from pulgon_tools.Irreps_tables_withparities import (
    line_group_sympy_withparities,
    sym_inverse_eye,
)
from pulgon_tools.utils import (
    get_character_num,
    get_character_num_withparities,
    get_character_withparities,
)

pytest_plugins = ["pytest-datadir"]


class TestLineGroupSymmetryDataset:
    """Test get_linegroup_symmetry_dataset returns correct symmetry info."""

    def test_dataset_family4_9_9_AM(self, shared_datadir):
        atom = read_vasp(shared_datadir / "9-9-AM")
        _, family, nrot, aL, ops, _ = get_linegroup_symmetry_dataset(atom)
        assert family == 4
        assert nrot == 9
        assert np.isclose(aL, 3.2125, atol=1e-3)
        assert len(ops) == 18

    def test_dataset_family4_12_12_AM(self, shared_datadir):
        atom = read_vasp(shared_datadir / "12-12-AM")
        _, family, nrot, aL, ops, _ = get_linegroup_symmetry_dataset(atom)
        assert family == 4
        assert nrot == 12
        assert np.isclose(aL, 3.192, atol=1e-3)
        assert len(ops) == 24

    def test_dataset_family8_10_0_ZZ(self, shared_datadir):
        atom = read_vasp(shared_datadir / "10-0-ZZ")
        _, family, nrot, aL, ops, _ = get_linegroup_symmetry_dataset(atom)
        assert family == 8
        assert nrot == 10
        assert np.isclose(aL, 5.5642, atol=1e-3)
        assert len(ops) == 40

    def test_dataset_family8_24_0_ZZ(self, shared_datadir):
        atom = read_vasp(shared_datadir / "24-0-ZZ")
        _, family, nrot, aL, ops, _ = get_linegroup_symmetry_dataset(atom)
        assert family == 8
        assert nrot == 24
        assert np.isclose(aL, 5.48, atol=1e-2)
        assert len(ops) == 144

    def test_dataset_family5_st1(self, shared_datadir):
        atom = read_vasp(shared_datadir / "st1")
        _, family, nrot, aL, ops, _ = get_linegroup_symmetry_dataset(atom)
        assert family == 5
        assert nrot == 8
        assert np.isclose(aL, 4.5, atol=1e-2)
        assert len(ops) == 28

    def test_dataset_accepts_atoms_object(self, shared_datadir):
        atom = read_vasp(shared_datadir / "10-0-ZZ")
        _, family, nrot, _, _, _ = get_linegroup_symmetry_dataset(atom)
        assert family == 8
        assert nrot == 10


class TestCharacterTable:
    """Test character table computation and orthogonality."""

    def _get_dict_params(self, shared_datadir, name, qz=0.0):
        atom = read_vasp(shared_datadir / name)
        _, family, nrot, aL, _, order_ops = get_linegroup_symmetry_dataset(
            atom
        )
        qp_normalized = qz / aL * 2 * np.pi
        return {
            "qpoints": qp_normalized,
            "nrot": nrot,
            "order": order_ops,
            "family": family,
            "a": aL,
        }

    def test_character_shape_family8(self, shared_datadir):
        params = self._get_dict_params(shared_datadir, "10-0-ZZ")
        characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
        assert characters.shape == (13, 40)

    def test_character_shape_family4(self, shared_datadir):
        params = self._get_dict_params(shared_datadir, "9-9-AM")
        characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
        assert characters.shape == (36, 18)

    def test_orthogonality_family8_q0(self, shared_datadir):
        """Character orthogonality: sum_g chi_i(g)* chi_j(g) / |G| = delta_ij."""
        params = self._get_dict_params(shared_datadir, "10-0-ZZ", qz=0.0)
        characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
        G = characters.shape[1]
        orth = characters @ characters.conj().T / G
        assert np.allclose(np.diag(orth).real, 1.0, atol=1e-10)
        off_diag = orth - np.diag(np.diag(orth))
        assert np.allclose(off_diag, 0.0, atol=1e-10)

    def test_orthogonality_family8_qhalf(self, shared_datadir):
        params = self._get_dict_params(shared_datadir, "10-0-ZZ", qz=0.5)
        characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
        G = characters.shape[1]
        orth = characters @ characters.conj().T / G
        assert np.allclose(np.diag(orth).real, 1.0, atol=1e-10)
        off_diag = orth - np.diag(np.diag(orth))
        assert np.allclose(off_diag, 0.0, atol=1e-10)

    def test_identity_character_equals_dimension(self, shared_datadir):
        """The character of the identity operation equals the irrep dimension."""
        params = self._get_dict_params(shared_datadir, "10-0-ZZ")
        characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
        rep_mat, _, _ = get_character_withparities(params, symprec=1e-8)
        for i, rep in enumerate(rep_mat):
            if rep.ndim == 1:
                assert np.isclose(characters[i, 0].real, 1.0)
            else:
                dim = rep.shape[1]
                assert np.isclose(characters[i, 0].real, dim)

    def test_unsupported_family_raises(self, shared_datadir):
        params = self._get_dict_params(shared_datadir, "st1")
        with pytest.raises(NotImplementedError):
            get_character_num_withparities(params, symprec=1e-8)


class TestRepresentationMatrices:
    """Test irreducible representation matrices."""

    def _get_dict_params(self, shared_datadir, name, qz=0.0):
        atom = read_vasp(shared_datadir / name)
        _, family, nrot, aL, _, order_ops = get_linegroup_symmetry_dataset(
            atom
        )
        qp_normalized = qz / aL * 2 * np.pi
        return {
            "qpoints": qp_normalized,
            "nrot": nrot,
            "order": order_ops,
            "family": family,
            "a": aL,
        }

    def test_rep_matrix_count(self, shared_datadir):
        params = self._get_dict_params(shared_datadir, "10-0-ZZ")
        rep_mat, _, _ = get_character_withparities(params, symprec=1e-8)
        assert len(rep_mat) == 13

    def test_rep_matrix_dimensions(self, shared_datadir):
        """Check that 1D irreps are vectors and 2D irreps are (nops, 2, 2)."""
        params = self._get_dict_params(shared_datadir, "10-0-ZZ")
        rep_mat, _, _ = get_character_withparities(params, symprec=1e-8)
        nops = 40
        for rep in rep_mat:
            if rep.ndim == 1:
                assert rep.shape == (nops,)
            else:
                assert rep.shape == (nops, 2, 2)

    def test_rep_trace_equals_character(self, shared_datadir):
        """Trace of representation matrix should equal the character."""
        params = self._get_dict_params(shared_datadir, "10-0-ZZ")
        rep_mat, _, _ = get_character_withparities(params, symprec=1e-8)
        characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
        for i, rep in enumerate(rep_mat):
            if rep.ndim == 1:
                assert np.allclose(rep, characters[i], atol=1e-10)
            else:
                traces = np.trace(rep, axis1=1, axis2=2)
                assert np.allclose(traces, characters[i], atol=1e-10)

    def test_rep_matrix_family4(self, shared_datadir):
        params = self._get_dict_params(shared_datadir, "12-12-AM")
        rep_mat, _, _ = get_character_withparities(params, symprec=1e-8)
        assert len(rep_mat) == 48
        nops = 24
        for rep in rep_mat:
            if rep.ndim == 1:
                assert rep.shape == (nops,)
            else:
                assert rep.shape[0] == nops


def _make_params(shared_datadir, name, qz=0.0):
    """Helper to build DictParams from a structure name."""
    atom = read_vasp(shared_datadir / name)
    _, family, nrot, aL, _, order_ops = get_linegroup_symmetry_dataset(atom)
    qp_normalized = qz / aL * 2 * np.pi
    return {
        "qpoints": qp_normalized,
        "nrot": nrot,
        "order": order_ops,
        "family": family,
        "a": aL,
    }


class TestFracRange:
    """Test frac_range utility from Irreps_tables."""

    def test_basic_range(self):
        assert frac_range(0.5, 3.5) == [1, 2, 3]

    def test_inclusive_boundaries(self):
        assert frac_range(1.0, 3.0) == [1, 2, 3]

    def test_exclude_left(self):
        assert frac_range(1.0, 3.0, left=False) == [2, 3]

    def test_exclude_right(self):
        assert frac_range(1.0, 3.0, right=False) == [1, 2]

    def test_exclude_both(self):
        assert frac_range(1.0, 3.0, left=False, right=False) == [2]

    def test_non_integer_boundaries(self):
        assert frac_range(1.1, 2.9) == [2]


class TestSymInverseEye:
    """Test sym_inverse_eye from Irreps_tables_withparities."""

    def test_size_2(self):
        M = sym_inverse_eye(2)
        expected = sympy.Matrix([[0, 1], [1, 0]])
        assert M == expected

    def test_size_3(self):
        M = sym_inverse_eye(3)
        expected = sympy.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert M == expected

    def test_involution(self):
        """Anti-diagonal identity squared should be the identity."""
        M = sym_inverse_eye(4)
        assert M * M == sympy.eye(4)


class TestLineGroupSympyNoParities:
    """Test line_group_sympy (without parities) for different families."""

    def test_family4_returns_characters(self, shared_datadir):
        params = _make_params(shared_datadir, "9-9-AM")
        rep_mat, vals, syms = line_group_sympy(params, symprec=1e-8)
        assert len(rep_mat) == 18
        assert len(vals) == 18

    def test_family6_returns_characters(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v")
        rep_mat, vals, syms = line_group_sympy(params, symprec=1e-8)
        assert len(rep_mat) == 3
        assert len(vals) == 3

    def test_family8_returns_characters(self, shared_datadir):
        params = _make_params(shared_datadir, "10-0-ZZ")
        rep_mat, vals, syms = line_group_sympy(params, symprec=1e-8)
        assert len(rep_mat) == 11

    def test_family6_nonzero_qpoint(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v", qz=0.3)
        rep_mat, vals, syms = line_group_sympy(params, symprec=1e-8)
        assert len(rep_mat) > 0

    def test_family4_nonzero_qpoint(self, shared_datadir):
        params = _make_params(shared_datadir, "9-9-AM", qz=0.3)
        rep_mat, vals, syms = line_group_sympy(params, symprec=1e-8)
        assert len(rep_mat) > 0

    def test_family8_nonzero_qpoint(self, shared_datadir):
        params = _make_params(shared_datadir, "10-0-ZZ", qz=0.3)
        rep_mat, vals, syms = line_group_sympy(params, symprec=1e-8)
        assert len(rep_mat) > 0


class TestCharacterNumNoParities:
    """Test get_character_num (traces from line_group_sympy)."""

    def test_family6_shape(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v")
        chars, vals, syms = get_character_num(params, symprec=1e-8)
        assert chars.shape == (3, 8)

    def test_family4_shape(self, shared_datadir):
        params = _make_params(shared_datadir, "9-9-AM")
        chars, _, _ = get_character_num(params, symprec=1e-8)
        assert chars.shape == (18, 18)

    def test_family8_shape(self, shared_datadir):
        params = _make_params(shared_datadir, "10-0-ZZ")
        chars, _, _ = get_character_num(params, symprec=1e-8)
        assert chars.shape == (11, 40)

    def test_family6_orthogonality(self, shared_datadir):
        """Off-diagonal elements of character inner product should vanish."""
        params = _make_params(shared_datadir, "C4v")
        chars, _, _ = get_character_num(params, symprec=1e-8)
        G = chars.shape[1]
        orth = chars @ chars.conj().T / G
        off_diag = orth - np.diag(np.diag(orth))
        assert np.allclose(off_diag, 0.0, atol=1e-10)

    def test_family8_orthogonality(self, shared_datadir):
        params = _make_params(shared_datadir, "10-0-ZZ")
        chars, _, _ = get_character_num(params, symprec=1e-8)
        G = chars.shape[1]
        orth = chars @ chars.conj().T / G
        off_diag = orth - np.diag(np.diag(orth))
        assert np.allclose(off_diag, 0.0, atol=1e-10)


class TestWithParitiesAdditionalFamilies:
    """Test line_group_sympy_withparities for families beyond 4 and 8."""

    def test_family6_q0(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v")
        rep_mat, vals, syms = line_group_sympy_withparities(
            params, symprec=1e-8
        )
        assert len(rep_mat) == 5

    def test_family6_nonzero_q(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v", qz=0.3)
        rep_mat, vals, syms = line_group_sympy_withparities(
            params, symprec=1e-8
        )
        assert len(rep_mat) > 0

    def test_family6_orthogonality(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v")
        chars, _, _ = get_character_num_withparities(params, symprec=1e-8)
        G = chars.shape[1]
        orth = chars @ chars.conj().T / G
        assert np.allclose(np.diag(orth).real, 1.0, atol=1e-10)
        off_diag = orth - np.diag(np.diag(orth))
        assert np.allclose(off_diag, 0.0, atol=1e-10)

    def test_family4_nonzero_q(self, shared_datadir):
        params = _make_params(shared_datadir, "9-9-AM", qz=0.3)
        rep_mat, vals, syms = line_group_sympy_withparities(
            params, symprec=1e-8
        )
        assert len(rep_mat) > 0

    def test_family8_24_0_ZZ(self, shared_datadir):
        """Larger family 8 structure to exercise more code paths."""
        params = _make_params(shared_datadir, "24-0-ZZ")
        chars, _, _ = get_character_num_withparities(params, symprec=1e-8)
        assert chars.shape == (27, 144)


# --- Manual DictParams for families without test structures ---

_FAMILY2_ORDER = [
    [0],
    [0, 1],
    [0, 1, 1],
    [0, 1, 1, 1],
    [2],
    [0, 2],
    [0, 1, 2],
    [0, 1, 1, 2],
]

_FAMILY3_ORDER = [
    [0],
    [0, 1],
    [0, 1, 1],
    [0, 1, 1, 1],
    [2],
    [0, 2],
    [0, 1, 2],
    [0, 1, 1, 2],
]

_FAMILY13_ORDER = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 4],
    [1, 1, 3],
    [1, 1, 4],
    [1, 2, 3],
    [1, 2, 4],
]


class TestLineGroupSympyFamily2:
    """Test line_group_sympy for family 2 (S2n groups)."""

    def _params(self, qpoints=0.0):
        return {
            "family": 2,
            "nrot": 4,
            "qpoints": qpoints,
            "a": 3.0,
            "order": _FAMILY2_ORDER,
        }

    def test_q0_returns_1d_chars(self):
        chars, vals, syms = line_group_sympy(self._params(0.0), symprec=1e-6)
        assert len(chars) == 4
        for c in chars:
            assert np.array(c).shape == (8,)

    def test_boundary_q_returns_1d_chars(self):
        a = 3.0
        chars, vals, _ = line_group_sympy(
            self._params(np.pi / a), symprec=1e-6
        )
        assert len(chars) == 4
        for c in chars:
            assert np.array(c).shape == (8,)

    def test_nonzero_q_returns_2d_matrices(self):
        chars, vals, _ = line_group_sympy(self._params(0.5), symprec=1e-6)
        assert len(chars) == 4
        for c in chars:
            assert np.array(c).shape == (8, 2, 2)


class TestWithParitiesFamily2:
    """Test line_group_sympy_withparities for family 2 (S2n groups)."""

    def _params(self, qpoints=0.0):
        return {
            "family": 2,
            "nrot": 4,
            "qpoints": qpoints,
            "a": 3.0,
            "order": _FAMILY2_ORDER,
        }

    def test_q0_rep_count(self):
        """At q=0, family 2 produces 2*nrot 1D irreps (piH = +/-1)."""
        reps, vals, syms = line_group_sympy_withparities(
            self._params(0.0), symprec=1e-6
        )
        assert len(reps) == 8

    def test_boundary_q_rep_count(self):
        a = 3.0
        reps, vals, _ = line_group_sympy_withparities(
            self._params(np.pi / a), symprec=1e-6
        )
        assert len(reps) == 8

    def test_nonzero_q_returns_2d_matrices(self):
        reps, vals, _ = line_group_sympy_withparities(
            self._params(0.5), symprec=1e-6
        )
        assert len(reps) == 4
        for r in reps:
            assert np.array(r).shape == (8, 2, 2)

    def test_q0_piH_values_present(self):
        """Each (k, m) pair should generate two irreps with piH=-1 and +1."""
        _, vals, _ = line_group_sympy_withparities(
            self._params(0.0), symprec=1e-6
        )
        piH_values = [v[2] for v in vals]
        assert piH_values.count(-1) == 4
        assert piH_values.count(1) == 4


class TestLineGroupSympyFamily3:
    """Test line_group_sympy for family 3 (Cnh groups)."""

    def _params(self, qpoints=0.0):
        return {
            "family": 3,
            "nrot": 4,
            "qpoints": qpoints,
            "a": 3.0,
            "order": _FAMILY3_ORDER,
        }

    def test_q0_returns_characters(self):
        chars, vals, syms = line_group_sympy(self._params(0.0), symprec=1e-6)
        assert len(chars) == 4

    def test_boundary_q_returns_characters(self):
        a = 3.0
        chars, _, _ = line_group_sympy(self._params(np.pi / a), symprec=1e-6)
        assert len(chars) == 4

    def test_nonzero_q_returns_2d_matrices(self):
        chars, _, _ = line_group_sympy(self._params(0.5), symprec=1e-6)
        assert len(chars) == 4
        for c in chars:
            assert np.array(c).shape == (8, 2, 2)


class TestLineGroupSympyFamily13:
    """Test line_group_sympy for family 13 (Dnd groups)."""

    def _params(self, qpoints=0.0):
        return {
            "family": 13,
            "nrot": 2,
            "qpoints": qpoints,
            "a": 3.0,
            "order": _FAMILY13_ORDER,
        }

    def test_q0_mixed_dimensions(self):
        """At q=0 family 13 produces both 1D and 2D irreps."""
        chars, vals, _ = line_group_sympy(self._params(0.0), symprec=1e-6)
        assert len(chars) == 3
        dims = [np.array(c).ndim for c in chars]
        assert 1 in dims and 3 in dims

    def test_boundary_q(self):
        a = 3.0
        chars, vals, _ = line_group_sympy(
            self._params(np.pi / a), symprec=1e-6
        )
        assert len(chars) == 2

    def test_nonzero_q_has_4d_irreps(self):
        """Interior q produces 4x4 matrices for some irreps."""
        chars, vals, _ = line_group_sympy(self._params(0.5), symprec=1e-6)
        shapes = [np.array(c).shape for c in chars]
        assert any(s[-1] == 4 for s in shapes if len(s) == 3)


class TestWithParitiesFamily13:
    """Test line_group_sympy_withparities for family 13."""

    def _params(self, qpoints=0.0):
        return {
            "family": 13,
            "nrot": 2,
            "qpoints": qpoints,
            "a": 3.0,
            "order": _FAMILY13_ORDER,
        }

    def test_q0_rep_count(self):
        reps, vals, syms = line_group_sympy_withparities(
            self._params(0.0), symprec=1e-6
        )
        assert len(reps) == 10

    def test_boundary_q_rep_count(self):
        a = 3.0
        reps, vals, _ = line_group_sympy_withparities(
            self._params(np.pi / a), symprec=1e-6
        )
        assert len(reps) == 4

    def test_nonzero_q_rep_count(self):
        reps, vals, _ = line_group_sympy_withparities(
            self._params(0.5), symprec=1e-6
        )
        assert len(reps) == 5


class TestWithParitiesFamily6Boundary:
    """Test withparities at boundary q-point for family 6."""

    def test_boundary_q_rep_count(self, shared_datadir):
        params = _make_params(shared_datadir, "C4v", qz=0.5)
        # q=0.5/aL * 2pi is in interior
        reps, vals, _ = line_group_sympy_withparities(params, symprec=1e-8)
        assert len(reps) > 0

    def test_boundary_q_orthogonality(self, shared_datadir):
        """Character orthogonality at q=pi/a for family 6."""
        atom = read_vasp(shared_datadir / "C4v")
        _, family, nrot, aL, _, order_ops = get_linegroup_symmetry_dataset(
            atom
        )
        params = {
            "qpoints": np.pi / aL,
            "nrot": nrot,
            "order": order_ops,
            "family": family,
            "a": aL,
        }
        chars, _, _ = get_character_num_withparities(params, symprec=1e-8)
        G = chars.shape[1]
        orth = chars @ chars.conj().T / G
        assert np.allclose(np.diag(orth).real, 1.0, atol=1e-10)
