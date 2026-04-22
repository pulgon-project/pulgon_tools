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
import phonopy
import pytest
from ase import Atoms

from pulgon_tools.symmetry_projector import (
    get_adapted_matrix_withparities,
    get_linegroup_symmetry_dataset,
)
from pulgon_tools.utils import get_matrices_withPhase

pytest_plugins = ["pytest-datadir"]

SP_DATA = "test/data/symmetry_projector"
SP_DATA_SC10 = "test/data/symmetry_projector_C_nanotube-sp10"


@pytest.fixture
def phonon_obj():
    """Load the phonopy object from symmetry_projector test data."""
    return phonopy.load(
        phonopy_yaml=f"{SP_DATA}/phonopy.yaml",
        force_constants_filename=f"{SP_DATA}/FORCE_CONSTANTS",
    )


@pytest.fixture
def symmetry_dataset(phonon_obj):
    """Build symmetry dataset from phonon primitive cell."""
    prim = phonon_obj.primitive
    atom = Atoms(
        cell=prim.cell, numbers=prim.numbers, positions=prim.positions
    )
    (
        atom_center,
        family,
        nrot,
        aL,
        ops_car_sym,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(atom)
    return {
        "atom_center": atom_center,
        "family": family,
        "nrot": nrot,
        "aL": aL,
        "ops_car_sym": ops_car_sym,
        "order_ops": order_ops,
        "gen_angles": gen_angles,
        "num_atom": len(prim.numbers),
    }


@pytest.fixture
def adapted_result_gamma(symmetry_dataset):
    """Run get_adapted_matrix_withparities at Gamma point."""
    ds = symmetry_dataset
    qp_1dim = 0.0
    DictParams = {
        "qpoints": qp_1dim,
        "nrot": ds["nrot"],
        "order": ds["order_ops"],
        "family": ds["family"],
        "a": ds["aL"],
        **ds["gen_angles"],
    }
    matrices = get_matrices_withPhase(
        ds["atom_center"], ds["ops_car_sym"], qp_1dim, symprec=1e-3
    )
    (
        adapted,
        dimensions,
        paras_values,
        paras_symbols,
    ) = get_adapted_matrix_withparities(DictParams, ds["num_atom"], matrices)
    return adapted, dimensions, paras_values, paras_symbols, ds


@pytest.fixture
def adapted_result_nonzero_q(symmetry_dataset):
    """Run get_adapted_matrix_withparities at a non-zero q-point."""
    ds = symmetry_dataset
    qp_1dim = 0.25 * 2 * np.pi / ds["aL"]
    DictParams = {
        "qpoints": qp_1dim,
        "nrot": ds["nrot"],
        "order": ds["order_ops"],
        "family": ds["family"],
        "a": ds["aL"],
        **ds["gen_angles"],
    }
    matrices = get_matrices_withPhase(
        ds["atom_center"], ds["ops_car_sym"], qp_1dim, symprec=1e-3
    )
    (
        adapted,
        dimensions,
        paras_values,
        paras_symbols,
    ) = get_adapted_matrix_withparities(DictParams, ds["num_atom"], matrices)
    return adapted, dimensions, paras_values, paras_symbols, ds


@pytest.fixture
def adapted_result_bz_boundary(symmetry_dataset):
    """Run get_adapted_matrix_withparities at BZ boundary q = pi/a."""
    ds = symmetry_dataset
    qp_1dim = 0.5 * 2 * np.pi / ds["aL"]
    DictParams = {
        "qpoints": qp_1dim,
        "nrot": ds["nrot"],
        "order": ds["order_ops"],
        "family": ds["family"],
        "a": ds["aL"],
        **ds["gen_angles"],
    }
    matrices = get_matrices_withPhase(
        ds["atom_center"], ds["ops_car_sym"], qp_1dim, symprec=1e-3
    )
    (
        adapted,
        dimensions,
        paras_values,
        paras_symbols,
    ) = get_adapted_matrix_withparities(DictParams, ds["num_atom"], matrices)
    return adapted, dimensions, paras_values, paras_symbols, ds


class TestGetAdaptedMatrixWithParities:
    """Test the get_adapted_matrix_withparities function."""

    def test_returns_four_elements(self, adapted_result_gamma):
        """Function returns adapted, dimensions, paras_values,
        paras_symbols."""
        (
            adapted,
            dimensions,
            paras_values,
            paras_symbols,
            _,
        ) = adapted_result_gamma
        assert isinstance(adapted, np.ndarray)
        assert isinstance(dimensions, list)
        assert isinstance(paras_values, list)
        assert isinstance(paras_symbols, list)

    def test_adapted_first_dim_matches_ndof(self, adapted_result_gamma):
        """First dimension of adapted equals 3 * num_atom."""
        adapted, _, _, _, ds = adapted_result_gamma
        ndof = 3 * ds["num_atom"]
        assert adapted.shape[0] == ndof

    def test_dimensions_sum(self, adapted_result_gamma):
        """Sum of dimensions equals the number of columns in adapted."""
        adapted, dimensions, _, _, _ = adapted_result_gamma
        assert sum(dimensions) == adapted.shape[1]

    def test_dimensions_all_positive(self, adapted_result_gamma):
        """All dimensions should be positive integers."""
        _, dimensions, _, _, _ = adapted_result_gamma
        for d in dimensions:
            assert isinstance(d, (int, np.integer))
            assert d > 0

    def test_each_block_columns_orthonormal(self, adapted_result_gamma):
        """Within each irrep block, columns should be orthonormal."""
        adapted, dimensions, _, _, _ = adapted_result_gamma
        start = 0
        for i, d in enumerate(dimensions):
            block = adapted[:, start : start + d]
            overlap = block.conj().T @ block
            np.testing.assert_allclose(
                overlap,
                np.eye(d),
                atol=1e-8,
                err_msg=f"Block {i} columns not orthonormal",
            )
            start += d

    def test_paras_values_length_matches_num_irreps(
        self, adapted_result_gamma
    ):
        """paras_values should have one entry per irrep."""
        _, dimensions, paras_values, _, _ = adapted_result_gamma
        assert len(paras_values) == len(dimensions)

    def test_paras_symbols_non_empty(self, adapted_result_gamma):
        """paras_symbols should be a non-empty list of labels."""
        _, _, _, paras_symbols, _ = adapted_result_gamma
        assert len(paras_symbols) > 0
        for s in paras_symbols:
            assert str(s)  # convertible to string

    def test_nonzero_q_adapted_shape(self, adapted_result_nonzero_q):
        """Adapted matrix at non-zero q: first dim equals ndof."""
        adapted, _, _, _, ds = adapted_result_nonzero_q
        ndof = 3 * ds["num_atom"]
        assert adapted.shape[0] == ndof

    def test_nonzero_q_adapted_square(self, adapted_result_nonzero_q):
        """Adapted matrix at non-zero q should be square (ndof x ndof)."""
        adapted, _, _, _, ds = adapted_result_nonzero_q
        ndof = 3 * ds["num_atom"]
        assert adapted.shape == (ndof, ndof)

    def test_nonzero_q_dimensions_sum(self, adapted_result_nonzero_q):
        """Sum of dimensions equals the number of columns."""
        adapted, dimensions, _, _, _ = adapted_result_nonzero_q
        assert sum(dimensions) == adapted.shape[1]

    def test_nonzero_q_dimensions_all_positive(self, adapted_result_nonzero_q):
        """All dimensions at non-zero q should be positive integers."""
        _, dimensions, _, _, _ = adapted_result_nonzero_q
        for d in dimensions:
            assert isinstance(d, (int, np.integer))
            assert d > 0

    def test_nonzero_q_each_block_orthonormal(self, adapted_result_nonzero_q):
        """Within each irrep block at non-zero q, columns should be
        orthonormal."""
        adapted, dimensions, _, _, _ = adapted_result_nonzero_q
        start = 0
        for i, d in enumerate(dimensions):
            block = adapted[:, start : start + d]
            overlap = block.conj().T @ block
            np.testing.assert_allclose(
                overlap,
                np.eye(d),
                atol=1e-8,
                err_msg=f"Block {i} columns not orthonormal",
            )
            start += d

    def test_nonzero_q_paras_values_length(self, adapted_result_nonzero_q):
        """paras_values should have one entry per irrep at non-zero q."""
        _, dimensions, paras_values, _, _ = adapted_result_nonzero_q
        assert len(paras_values) == len(dimensions)

    def test_nonzero_q_transforms_dynamical_matrix(
        self, adapted_result_nonzero_q, phonon_obj
    ):
        """Transformed D at non-zero q should be Hermitian."""
        adapted, dimensions, _, _, _ = adapted_result_nonzero_q
        q = [0.0, 0.0, 0.25]
        dmat = phonon_obj.get_dynamical_matrix_at_q(q)
        D_adapted = adapted.conj().T @ dmat @ adapted
        np.testing.assert_allclose(D_adapted, D_adapted.conj().T, atol=1e-8)

    def test_adapted_transforms_dynamical_matrix(
        self, adapted_result_gamma, phonon_obj
    ):
        """Transformed dynamical matrix D' = U^H D U should be
        Hermitian."""
        adapted, dimensions, _, _, _ = adapted_result_gamma
        q = [0.0, 0.0, 0.0]
        dmat = phonon_obj.get_dynamical_matrix_at_q(q)

        D_adapted = adapted.conj().T @ dmat @ adapted

        # D' should be Hermitian
        np.testing.assert_allclose(D_adapted, D_adapted.conj().T, atol=1e-8)

    def test_bz_boundary_adapted_square(self, adapted_result_bz_boundary):
        """Adapted matrix at BZ boundary should be square (ndof x ndof)."""
        adapted, _, _, _, ds = adapted_result_bz_boundary
        ndof = 3 * ds["num_atom"]
        assert adapted.shape == (ndof, ndof)

    def test_bz_boundary_dimensions_sum(self, adapted_result_bz_boundary):
        """Sum of dimensions equals the number of columns."""
        adapted, dimensions, _, _, _ = adapted_result_bz_boundary
        assert sum(dimensions) == adapted.shape[1]

    def test_bz_boundary_each_block_orthonormal(
        self, adapted_result_bz_boundary
    ):
        """Within each irrep block at BZ boundary, columns should be
        orthonormal."""
        adapted, dimensions, _, _, _ = adapted_result_bz_boundary
        start = 0
        for i, d in enumerate(dimensions):
            block = adapted[:, start : start + d]
            overlap = block.conj().T @ block
            np.testing.assert_allclose(
                overlap,
                np.eye(d),
                atol=1e-8,
                err_msg=f"Block {i} columns not orthonormal",
            )
            start += d

    def test_bz_boundary_transforms_dynamical_matrix(
        self, adapted_result_bz_boundary, phonon_obj
    ):
        """Transformed D at BZ boundary should be Hermitian."""
        adapted, dimensions, _, _, _ = adapted_result_bz_boundary
        q = [0.0, 0.0, 0.5]
        dmat = phonon_obj.get_dynamical_matrix_at_q(q)
        D_adapted = adapted.conj().T @ dmat @ adapted
        np.testing.assert_allclose(D_adapted, D_adapted.conj().T, atol=1e-8)


class TestSupercellAdapted:
    """Test symmetry projection on a 10x supercell at folded q-points."""

    @pytest.fixture
    def sc10_phonon(self):
        """Load the 10x supercell phonopy object."""
        return phonopy.load(
            phonopy_yaml=f"{SP_DATA_SC10}/phonopy.yaml",
            force_constants_filename=f"{SP_DATA_SC10}/FORCE_CONSTANTS",
        )

    @pytest.fixture
    def sc10_dataset(self, sc10_phonon):
        """Build symmetry dataset for 10x supercell."""
        prim = sc10_phonon.primitive
        atom = Atoms(
            cell=prim.cell,
            numbers=prim.numbers,
            positions=prim.positions,
        )
        ds = get_linegroup_symmetry_dataset(atom)
        atom_center, family, nrot, aL, ops, order, angles = ds
        return {
            "atom_center": atom_center,
            "family": family,
            "nrot": nrot,
            "aL": aL,
            "ops_car_sym": ops,
            "order_ops": order,
            "gen_angles": angles,
            "num_atom": len(prim.numbers),
        }

    @pytest.fixture
    def sc10_adapted_q01(self, sc10_dataset):
        """Adapted projection at q=0.1 (Gamma-point Bloch phase)."""
        ds = sc10_dataset
        qp_irrep = 0.1 * 2 * np.pi / ds["aL"]
        DictParams = {
            "qpoints": qp_irrep,
            "nrot": ds["nrot"],
            "order": ds["order_ops"],
            "family": ds["family"],
            "a": ds["aL"],
            **ds["gen_angles"],
        }
        # Bloch phase at Gamma (supercell calculation)
        qp_1dim = 0.0
        matrices = get_matrices_withPhase(
            ds["atom_center"],
            ds["ops_car_sym"],
            qp_1dim,
            symprec=1e-3,
        )
        adapted, dims, pvals, psyms = get_adapted_matrix_withparities(
            DictParams, ds["num_atom"], matrices
        )
        return adapted, dims, pvals, psyms, ds

    def test_sc10_adapted_shape(self, sc10_adapted_q01):
        """At q=0.1 the supercell (ndof=600) projects onto 60 modes
        (= 3 * 20 primitive-cell atoms)."""
        adapted, dims, _, _, ds = sc10_adapted_q01
        ndof = 3 * ds["num_atom"]
        assert adapted.shape == (ndof, 60)

    def test_sc10_dimensions_sum(self, sc10_adapted_q01):
        """Sum of irrep dimensions equals number of selected modes."""
        adapted, dims, _, _, _ = sc10_adapted_q01
        assert sum(dims) == adapted.shape[1]

    def test_sc10_each_block_orthonormal(self, sc10_adapted_q01):
        """Within each irrep block, columns should be orthonormal."""
        adapted, dims, _, _, _ = sc10_adapted_q01
        start = 0
        for i, d in enumerate(dims):
            block = adapted[:, start : start + d]
            overlap = block.conj().T @ block
            np.testing.assert_allclose(
                overlap,
                np.eye(d),
                atol=1e-8,
                err_msg=f"Block {i} columns not orthonormal",
            )
            start += d

    def test_sc10_transforms_dynamical_matrix(
        self, sc10_adapted_q01, sc10_phonon
    ):
        """Transformed D at q=0.1 should be Hermitian."""
        adapted, _, _, _, _ = sc10_adapted_q01
        q = [0.0, 0.0, 0.0]
        dmat = sc10_phonon.get_dynamical_matrix_at_q(q)
        D_adapted = adapted.conj().T @ dmat @ adapted
        np.testing.assert_allclose(D_adapted, D_adapted.conj().T, atol=1e-8)
