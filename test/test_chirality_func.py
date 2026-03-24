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

from pulgon_tools.structure_chirality import (
    bond_constraints_equations,
    cyl2car,
    generate_symcell_and_linegroup_elements,
    get_nanotube_from_n1n2,
    helical_group_analysis,
)


@pytest.fixture
def hex_lattice():
    """Standard MoS2 hexagonal lattice vectors
    from bond_length=2.43, delta_Z=1.57."""
    bond_length = 2.43
    delta_Z = 1.57
    L1 = np.sqrt(bond_length**2 - delta_Z**2) * np.sqrt(
        2 - 2 * np.cos(2 * np.pi / 3)
    )
    a1 = L1 * np.array([1, 0])
    a2 = np.array([L1 * np.cos(np.pi / 3), L1 * np.sin(np.pi / 3)])
    return a1, a2, L1, bond_length, delta_Z


class TestCyl2Car:
    """Test cylindrical to Cartesian coordinate conversion."""

    def test_basic_conversion(self):
        car = cyl2car([np.pi / 4, 2.0, 3.0])
        expected = np.array([np.sqrt(2), np.sqrt(2), 3.0])
        assert np.allclose(car, expected, atol=1e-10)

    def test_zero_angle(self):
        car = cyl2car([0, 5.0, 1.0])
        assert np.allclose(car, [5.0, 0.0, 1.0], atol=1e-10)

    def test_pi_half_angle(self):
        car = cyl2car([np.pi / 2, 3.0, 0.0])
        assert np.allclose(car, [0.0, 3.0, 0.0], atol=1e-10)

    def test_zero_radius(self):
        car = cyl2car([1.23, 0.0, 4.0])
        assert np.allclose(car, [0.0, 0.0, 4.0], atol=1e-10)


class TestHelicalGroupAnalysis:
    """Test helical group parameter computation."""

    def test_armchair_10_10(self, hex_lattice):
        a1, a2, L1, _, _ = hex_lattice
        q, f, r, R, Ch, t, t1, t2, n_gcd = helical_group_analysis(
            a1, a2, 10, 10, L1
        )
        assert q == 20
        assert n_gcd == 10
        assert t1 == -1 and t2 == 1
        assert R == 3
        assert np.isclose(t, 3.2125, atol=1e-3)
        assert np.isclose(r, 8.8557, atol=1e-3)

    def test_zigzag_10_0(self, hex_lattice):
        a1, a2, L1, _, _ = hex_lattice
        q, f, r, R, Ch, t, t1, t2, n_gcd = helical_group_analysis(
            a1, a2, 10, 0, L1
        )
        assert q == 20
        assert n_gcd == 10
        assert t1 == -1 and t2 == 2
        assert R == 3
        assert np.isclose(t, 5.5642, atol=1e-3)

    def test_chiral_6_3(self, hex_lattice):
        a1, a2, L1, _, _ = hex_lattice
        q, f, r, R, Ch, t, t1, t2, n_gcd = helical_group_analysis(
            a1, a2, 6, 3, L1
        )
        assert q == 42
        assert n_gcd == 3
        assert np.isclose(t, 14.7214, atol=1e-3)

    def test_q_equals_n_gcd_times_q_tilde(self, hex_lattice):
        """Verify q = n_gcd * q_tilde relationship."""
        a1, a2, L1, _, _ = hex_lattice
        for n1, n2 in [(10, 10), (10, 0), (9, 9), (6, 3)]:
            q, f, r, R, Ch, t, t1, t2, n_gcd = helical_group_analysis(
                a1, a2, n1, n2, L1
            )
            q_tilde = q // n_gcd
            assert np.isclose(f, t / q_tilde, atol=1e-10)

    def test_chiral_vector_is_unit(self, hex_lattice):
        a1, a2, L1, _, _ = hex_lattice
        _, _, _, _, Ch, _, _, _, _ = helical_group_analysis(a1, a2, 10, 10, L1)
        assert np.isclose(np.linalg.norm(Ch), 1.0, atol=1e-10)


class TestBondConstraintsEquations:
    """Test the bond constraint equations for fsolve."""

    def test_equal_distances_give_zero_eq1(self):
        """If distances to pos2 and pos3 are equal,
        eq1 (their difference) is zero."""
        r = 5.0
        pos1 = [0.0, r, 0.0]
        phi = 0.3
        pos2 = [phi, r, 0.0]
        pos3 = [-phi, r, 0.0]
        pos4 = [0.0, r, 1.0]
        residuals = bond_constraints_equations(
            [0, 0, 0], pos1, pos2, pos3, pos4, 2.0
        )
        # eq1 = dist(1,2) - dist(1,3) should be zero by symmetry
        assert np.isclose(residuals[0], 0.0, atol=1e-10)

    def test_returns_three_equations(self):
        pos = [0.0, 3.0, 0.0]
        result = bond_constraints_equations([0, 0, 0], pos, pos, pos, pos, 2.0)
        assert len(result) == 3


class TestGenerateSymcellAndLinegroupElements:
    """Test symmetry cell generation for MoS2-type nanotubes."""

    def test_symcell_has_three_atoms(self, hex_lattice):
        a1, a2, L1, bond_length, delta_Z = hex_lattice
        _, _, r, _, Ch, _, t1, t2, _ = helical_group_analysis(
            a1, a2, 10, 10, L1
        )
        pos_cyl, symbols = generate_symcell_and_linegroup_elements(
            a1,
            a2,
            Ch,
            t1,
            t2,
            r,
            bond_length,
            delta_Z,
            symbol1=42,
            symbol2=16,
        )
        assert pos_cyl.shape == (3, 3)
        assert len(symbols) == 3
        assert symbols[0] == 42
        assert symbols[1] == 16
        assert symbols[2] == 16

    def test_symcell_radii(self, hex_lattice):
        """Metal atom at r, chalcogens at r +/- delta_Z."""
        a1, a2, L1, bond_length, delta_Z = hex_lattice
        _, _, r, _, Ch, _, t1, t2, _ = helical_group_analysis(
            a1, a2, 10, 10, L1
        )
        pos_cyl, _ = generate_symcell_and_linegroup_elements(
            a1, a2, Ch, t1, t2, r, bond_length, delta_Z
        )
        # pos_cyl[:, 1] are the radii
        r_metal = pos_cyl[0, 1]
        assert np.isclose(r_metal, r, atol=1e-6)


class TestGetNanotubeFromN1N2:
    """Test full nanotube generation from chiral indices."""

    def test_armchair_10_10_atom_count(self):
        atom = get_nanotube_from_n1n2(10, 10, "Mo", "S", 2.43, 1.57)
        assert len(atom) == 60

    def test_zigzag_10_0_atom_count(self):
        atom = get_nanotube_from_n1n2(10, 0, "Mo", "S", 2.43, 1.57)
        assert len(atom) == 60

    def test_armchair_9_9_atom_count(self):
        atom = get_nanotube_from_n1n2(9, 9, "Mo", "S", 2.43, 1.57)
        assert len(atom) == 54

    def test_chiral_6_3_atom_count(self):
        atom = get_nanotube_from_n1n2(6, 3, "Mo", "S", 2.43, 1.57)
        assert len(atom) == 135

    def test_stoichiometry(self):
        """MoS2-type: n_chalcogen = 2 * n_metal."""
        atom = get_nanotube_from_n1n2(10, 10, "Mo", "S", 2.43, 1.57)
        symbols = atom.get_chemical_symbols()
        n_Mo = symbols.count("Mo")
        n_S = symbols.count("S")
        assert n_S == 2 * n_Mo

    def test_cell_z_equals_t(self):
        """Cell z-dimension should equal the translational period t."""
        atom = get_nanotube_from_n1n2(10, 10, "Mo", "S", 2.43, 1.57)
        assert np.isclose(atom.cell[2, 2], 3.2125, atol=1e-3)

    def test_bond_length_preserved(self):
        """Nearest Mo-S distance should match the input bond_length."""
        bond_length = 2.43
        atom = get_nanotube_from_n1n2(10, 10, "Mo", "S", bond_length, 1.57)
        pos = atom.positions
        symbols = atom.get_chemical_symbols()
        mo_idx = [i for i, s in enumerate(symbols) if s == "Mo"]
        s_idx = [i for i, s in enumerate(symbols) if s == "S"]
        min_d = min(
            np.linalg.norm(pos[i] - pos[j]) for i in mo_idx for j in s_idx
        )
        assert np.isclose(min_d, bond_length, atol=1e-3)

    def test_no_duplicate_atoms(self):
        """All atoms should be at unique positions."""
        atom = get_nanotube_from_n1n2(10, 10, "Mo", "S", 2.43, 1.57)
        scaled = atom.get_scaled_positions()
        unique = np.unique(np.round(scaled, 8), axis=0)
        assert len(unique) == len(atom)

    def test_different_symbols(self):
        """Support arbitrary element symbols."""
        atom = get_nanotube_from_n1n2(9, 9, "W", "Se", 2.43, 1.57)
        symbols = atom.get_chemical_symbols()
        assert "W" in symbols
        assert "Se" in symbols
        assert symbols.count("Se") == 2 * symbols.count("W")
