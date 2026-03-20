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
from ase.io.vasp import read_vasp

from pulgon_tools.generate_irreps_tables import get_linegroup_symmetry_dataset
from pulgon_tools.utils import (
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
