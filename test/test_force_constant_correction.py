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

import argparse

import numpy as np
import phonopy
import pytest
from ase import Atoms
from phonopy.file_IO import parse_FORCE_CONSTANTS

from pulgon_tools.force_constant_correction import (
    build_constraint_matrix,
    calc_dists,
    parse_bool_list,
    parse_int_list,
    solve_fcs,
    str2list,
)

pytest_plugins = ["pytest-datadir"]

FCS_DATA = "test/data/fcs"


@pytest.fixture
def phonon_obj():
    """Load the phonopy object from test data."""
    return phonopy.load(
        phonopy_yaml=f"{FCS_DATA}/phonopy.yaml",
        force_constants_filename=f"{FCS_DATA}/FORCE_CONSTANTS",
    )


@pytest.fixture
def supercell_atoms(phonon_obj):
    """Build an ASE Atoms object from the phonopy supercell."""
    scell = phonon_obj.supercell
    positions = (scell.scaled_positions % 1.0) @ scell.cell
    return Atoms(scell.symbols, positions, cell=scell.cell)


class TestCalcDists:
    """Test distance calculation with periodic images."""

    def test_dists_output(self, supercell_atoms):
        dists, degeneracy, shifts = calc_dists(supercell_atoms)
        n = len(supercell_atoms)
        min_d = dists[dists > 0.01].min()
        assert dists.shape == (n, n)
        assert degeneracy.shape == (n, n)
        assert shifts.shape[0] == n and shifts.shape[1] == n
        assert np.allclose(np.diag(dists), 0.0)
        assert 1.0 < min_d < 5.0
        assert (degeneracy >= 1).all()
        assert np.allclose(dists, dists.T, atol=1e-10)


class TestBuildConstraintMatrix:
    """Test constraint matrix construction."""

    def test_output_types(self, phonon_obj):
        M, IFC = build_constraint_matrix(phonon_obj)
        assert hasattr(M, "nnz")
        assert isinstance(IFC, np.ndarray)
        n_prim = len(phonon_obj.primitive)
        n_super = len(phonon_obj.supercell)
        assert IFC.shape == (n_prim, n_super, 3, 3)
        assert M.shape[1] == IFC.size
        assert M.nnz > 0

    def test_raw_ifc_violates_constraints(self, phonon_obj):
        """Raw IFC should have nonzero constraint violation."""
        M, IFC = build_constraint_matrix(phonon_obj)
        violation = np.abs(M.dot(IFC.ravel())).sum()
        assert violation > 1.0


class TestSolveFcs:
    """Test force constant correction solver."""

    @pytest.fixture
    def constraint_data(self, phonon_obj):
        M, IFC = build_constraint_matrix(phonon_obj)
        return M, IFC

    def test_convex_opt_satisfies_constraints(self, constraint_data):
        """After correction, M @ x should be nearly zero."""
        M, IFC = constraint_data
        IFC_sym = solve_fcs(IFC, M, methods="convex_opt")
        violation = np.abs(M.dot(IFC_sym.ravel())).sum()
        fcs_ref = parse_FORCE_CONSTANTS(
            f"{FCS_DATA}/FORCE_CONSTANTS_correction"
        )
        assert violation < 1e-6
        assert not np.allclose(IFC_sym, IFC)
        assert np.allclose(IFC_sym, fcs_ref, atol=0.15)

    def test_ridge_model_reduces_violation(self, constraint_data):
        """Ridge regression should reduce constraint violation."""
        M, IFC = constraint_data
        violation_before = np.abs(M.dot(IFC.ravel())).sum()
        IFC_sym = solve_fcs(IFC, M, methods="ridge_model")
        violation_after = np.abs(M.dot(IFC_sym.ravel())).sum()
        assert IFC_sym.shape == IFC.shape
        assert violation_after < violation_before


class TestParseIntList:
    """Test parse_int_list argument parser."""

    def test_valid_input(self):
        assert parse_int_list("[5, 5, 1]") == [5, 5, 1]

    def test_wrong_length(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_int_list("[1, 2]")

    def test_non_int(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_int_list("[1.0, 2.0, 3.0]")

    def test_invalid_string(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_int_list("not_a_list")


class TestParseBoolList:
    """Test parse_bool_list argument parser."""

    def test_valid_input(self):
        assert parse_bool_list("[True, True, False]") == [True, True, False]

    def test_wrong_length(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bool_list("[True, False]")

    def test_non_bool(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bool_list("[1, 0, 1]")


class TestStr2List:
    """Test str2list argument parser."""

    def test_valid_input(self):
        result = str2list("[[0, 0, 0], [0.5, 0, 0]]")
        assert result == [[0, 0, 0], [0.5, 0, 0]]

    def test_none_input(self):
        assert str2list(None) is None

    def test_invalid_string(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2list("not valid python")
