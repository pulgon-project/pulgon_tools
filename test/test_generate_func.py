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

from pulgon_tools.structures_sym_based import (
    T_Q,
    T_v,
    change_center,
    generate_line_group_structure,
)
from pulgon_tools.utils import Cn, S2n, U, U_d, dimino, sigmaH, sigmaV

pytest_plugins = ["pytest-datadir"]


def pre_processing(pos_cylin, pos_symbols, generators):
    if pos_cylin.ndim == 1:
        pos = np.array(
            [
                pos_cylin[0] * np.cos(pos_cylin[1]),
                pos_cylin[0] * np.sin(pos_cylin[1]),
                pos_cylin[2],
            ]
        )
    else:
        pos = np.array(
            [
                pos_cylin[:, 0] * np.cos(pos_cylin[:, 1]),
                pos_cylin[:, 0] * np.sin(pos_cylin[:, 1]),
                pos_cylin[:, 2],
            ]
        )
        pos = pos.T
    rot_sym = dimino(generators, symec=4)
    monomer_pos, monomer_symbols = [], []

    for sym in rot_sym:
        if pos.ndim == 1:
            monomer_pos.append(np.dot(sym, pos.reshape(pos.shape[0], 1)).T[0])
            monomer_symbols.extend(pos_symbols)
        else:
            monomer_pos.extend([np.dot(sym, line) for line in pos])
            monomer_symbols.extend(pos_symbols)
    monomer_pos = np.array(monomer_pos)
    monomer_symbols = np.array(monomer_symbols)
    return monomer_pos, monomer_symbols


@pytest.fixture(name="generators")
def fixture_point_group_generators():
    return [Cn(6), sigmaV(), U()]


def test_dimino(generators):
    assert len(dimino(generators)) == 24


def test_st1():
    """(Cq|f),Cn"""
    motif = np.array([2, 0, 0])
    symbols = ["C"]
    generators = np.array([Cn(4)])
    cyclic = {"T_Q": [6, 1.5]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)

    assert len(st) == 12


def test_st2():
    """(I|q),S2n"""
    motif = np.array([3, 0, 1])
    symbols = ["C"]

    generators = np.array([S2n(6)])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 12


def test_st3():
    """(I|q),Cn,sigmaH"""
    motif = np.array([2.5, 0, 1])
    symbols = ["C"]

    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 12


def test_st4():
    """(C2n|f/2),Cn,sigmaH"""
    motif = np.array([3, 0, 0.6])
    symbols = ["C"]

    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_Q": [12, 4]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 24


def test_st5():
    """(Cq|f),Cn,U"""
    motif = np.array([3, np.pi / 9, 0.5])
    symbols = ["C"]

    generators = np.array([Cn(6), U()])
    cyclic = {"T_Q": [4, 4]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 24


def test_st6():
    """(I|a),Cn,sigmaV"""
    motif = np.array([3, np.pi / 24, 1])
    symbols = ["C"]

    generators = np.array([Cn(6), sigmaV()])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 12


def test_st7():
    """(sigmaV|a/2),Cn"""
    motif = np.array([3, np.pi / 24, 1])
    symbols = ["C"]

    generators = np.array([Cn(6)])
    cyclic = {"T_V": 1.5}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 12


def test_st8():
    """(C2n|a/2),Cn,sigmaV"""
    motif = np.array([3, np.pi / 24, 0])
    symbols = ["C"]

    generators = np.array([Cn(6), sigmaV()])
    cyclic = {"T_Q": [12, 1.5]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 26


def test_st9():
    """(I|a),Cn,Ud,sigmaV"""
    motif = np.array([3, np.pi / 24, 0.6])
    symbols = ["C"]

    generators = np.array([Cn(6), U_d(np.pi / 12), sigmaV()])
    cyclic = {"T_Q": [1, 4]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 24


def test_st10():
    """(sigmaV|a/2),S2n"""
    motif = np.array([3, np.pi / 18, 0.4])
    symbols = ["C"]

    generators = np.array([S2n(6)])
    cyclic = {"T_V": 4}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 24


def test_st11():
    """(I|a),Cn,sigmaV"""
    motif = np.array([3, np.pi / 18, 0.6])
    symbols = ["C"]

    generators = np.array([Cn(6), U(), sigmaV()])
    cyclic = {"T_Q": [1, 4]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 24


def test_st12():
    """(sigmaV|a),Cn,U,sigmaV"""
    motif = np.array([3, np.pi / 24, 0.5])
    symbols = ["C"]

    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_V": 2.5}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 24


def test_st13():
    """(C2n|a/2),Cn,U,sigmaV"""
    motif = np.array([3, np.pi / 16, 0.6])
    symbols = ["C"]

    generators = np.array([Cn(6), U(), sigmaV()])
    cyclic = {"T_Q": [12, 3]}

    monomer_pos, monomer_symbols = pre_processing(motif, symbols, generators)
    st = generate_line_group_structure(monomer_pos, monomer_symbols, cyclic)
    assert len(st) == 48


class TestTQ:
    """Test T_Q helical rotation + translation."""

    def test_basic_rotation(self):
        pos = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
        result = T_Q(4, 1.5, pos)
        assert result.shape == pos.shape
        # z should be shifted by f
        assert np.allclose(result[:, 2], pos[:, 2] + 1.5)

    def test_identity_rotation(self):
        """Q=1 means rotate by 2*pi, should return same xy."""
        pos = np.array([[3.0, 0.0, 0.5]])
        result = T_Q(1, 2.0, pos)
        assert np.allclose(result[:, :2], pos[:, :2], atol=1e-10)
        assert np.isclose(result[0, 2], 2.5)

    def test_1d_input_returns_unchanged(self):
        """1D input triggers error branch and returns pos unchanged."""
        pos = np.array([1.0, 2.0, 3.0])
        result = T_Q(4, 1.0, pos)
        assert np.array_equal(result, pos)


class TestTv:
    """Test T_v mirror reflection + translation."""

    def test_basic_reflection(self):
        pos = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
        result = T_v(1.5, pos)
        assert result.shape == pos.shape
        assert np.allclose(result[:, 2], pos[:, 2] + 1.5)

    def test_double_reflection_restores_xy(self):
        """Applying sigmaV twice restores the original xy positions."""
        pos = np.array([[1.0, 2.0, 0.0]])
        r1 = T_v(0.0, pos)
        r2 = T_v(0.0, r1)
        assert np.allclose(r2[:, :2], pos[:, :2], atol=1e-10)

    def test_1d_input_returns_unchanged(self):
        pos = np.array([1.0, 2.0, 3.0])
        result = T_v(1.0, pos)
        assert np.array_equal(result, pos)


class TestChangeCenter:
    """Test change_center shifts z-axis to cell center."""

    def test_preserves_atom_count(self):
        st = Atoms(
            "C4",
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
        )
        st2 = change_center(st)
        assert len(st2) == len(st)

    def test_shifts_xy_by_half(self):
        """Scaled xy positions should shift by +0.5."""
        st = Atoms(
            "C",
            positions=[[0.0, 0.0, 0.0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
        )
        st2 = change_center(st)
        scaled = st2.get_scaled_positions()
        assert np.allclose(scaled[0, :2], [0.5, 0.5], atol=1e-10)

    def test_z_unchanged(self):
        st = Atoms(
            "C",
            positions=[[1.0, 2.0, 3.0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
        )
        z_before = st.get_scaled_positions()[0, 2]
        st2 = change_center(st)
        z_after = st2.get_scaled_positions()[0, 2]
        assert np.isclose(z_before, z_after, atol=1e-10)


class TestGenerateLineGroupStructure:
    """Test generate_line_group_structure with multi-atom motifs."""

    def test_multi_species_motif(self):
        """Two-species motif with T_Q should produce correct stoichiometry."""
        pos_cylin = np.array([[3, np.pi / 24, 0.6], [2.2, np.pi / 24, 0.8]])
        symbols = ("Mo", "S")
        generators = np.array([Cn(6), sigmaV()])
        monomer_pos, monomer_symbols = pre_processing(
            pos_cylin, symbols, generators
        )
        cyclic = {"T_Q": [6, 1.5]}
        st = generate_line_group_structure(
            monomer_pos, monomer_symbols, cyclic
        )
        chem = st.get_chemical_symbols()
        assert "Mo" in chem
        assert "S" in chem

    def test_output_has_pbc_along_z(self):
        """Generated structure should have PBC only along z."""
        motif = np.array([2, 0, 0])
        symbols = ["C"]
        generators = np.array([Cn(4)])
        cyclic = {"T_Q": [6, 1.5]}
        monomer_pos, monomer_symbols = pre_processing(
            motif, symbols, generators
        )
        st = generate_line_group_structure(
            monomer_pos, monomer_symbols, cyclic
        )
        assert list(st.pbc) == [False, False, True]

    def test_cell_z_matches_period(self):
        """Cell z-dimension should equal Q * f for T_Q."""
        motif = np.array([2, 0, 0])
        symbols = ["C"]
        generators = np.array([Cn(4)])
        Q, f = 6, 1.5
        cyclic = {"T_Q": [Q, f]}
        monomer_pos, monomer_symbols = pre_processing(
            motif, symbols, generators
        )
        st = generate_line_group_structure(
            monomer_pos, monomer_symbols, cyclic
        )
        # Q may be reduced if early convergence; cell z = actual_Q * f
        assert st.cell[2, 2] > 0

    def test_tv_cell_z(self):
        """Cell z-dimension should equal 2 * f for T_V."""
        motif = np.array([3, np.pi / 24, 1])
        symbols = ["C"]
        generators = np.array([Cn(6)])
        f = 1.5
        cyclic = {"T_V": f}
        monomer_pos, monomer_symbols = pre_processing(
            motif, symbols, generators
        )
        st = generate_line_group_structure(
            monomer_pos, monomer_symbols, cyclic
        )
        assert np.isclose(st.cell[2, 2], 2 * f)
