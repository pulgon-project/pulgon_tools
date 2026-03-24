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
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer, get_symcell

pytest_plugins = ["pytest-datadir"]


class TestCyclicGroupAnalyzer:
    def test_rotation(self, shared_datadir):
        st_name = shared_datadir / "st1"
        st = read_vasp(st_name)
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        monomers, translations = cy._potential_translation()
        idx, Q, _ = cy._detect_rotation(
            monomers[0], translations[0] * cy._primitive.cell[2, 2], 3
        )

        assert idx == True
        assert Q.numerator == 12

    def test_find_axis_center_of_nanotubFe(self, shared_datadir):
        st_name = shared_datadir / "12-12-AM"
        st = read_vasp(st_name)
        res = CyclicGroupAnalyzer(st)
        n_st = res._find_axis_center_of_nanotube(st)
        average_coord = (n_st.positions[:, :2] / len(n_st)).sum(axis=0)
        assert (average_coord - [0.5, 0.5] @ n_st.cell[:2, :2]).sum() < 0.001

    def test_generate_monomer(self, shared_datadir):
        st_name = shared_datadir / "9-9-AM"
        st = read_vasp(st_name)
        cy = CyclicGroupAnalyzer(st)
        monomers, translations = cy._potential_translation()
        assert str(monomers[0].symbols) == "Mo9S18"
        assert np.isclose(translations[0], 0.5, atol=1e-3)

    def test_rotational_tolerance(self, shared_datadir):
        st_name = shared_datadir / "st1"
        st = read_vasp(st_name)
        cy1 = CyclicGroupAnalyzer(st, tolerance=1e-2)
        cy2 = CyclicGroupAnalyzer(st, tolerance=1e-4)
        monomers1, translations1 = cy1._potential_translation()
        monomers2, translations2 = cy2._potential_translation()
        idx1, _, _ = cy1._detect_rotation(
            monomers1[0], translations1[0] * cy1._primitive.cell[2, 2], ind=3
        )
        idx2, _, _ = cy2._detect_rotation(
            monomers2[0], translations2[0] * cy2._primitive.cell[2, 2], ind=3
        )
        assert idx1 == True
        assert idx2 == False

    def test_mirror(self, shared_datadir):
        st_name = shared_datadir / "st7"
        st = read_vasp(st_name)
        cy = CyclicGroupAnalyzer(st)
        monomers, translations = cy._potential_translation()
        idx, _ = cy._detect_mirror(
            monomers[0], translations[0] * cy._primitive.cell[2, 2]
        )
        assert idx == True

    def test_mirror_tolerance(self, shared_datadir):
        st_name = shared_datadir / "st7"
        st = read_vasp(st_name)

        cy1 = CyclicGroupAnalyzer(st, tolerance=1e-14)
        cy2 = CyclicGroupAnalyzer(st, tolerance=1e-15)
        monomers1, translations1 = cy1._potential_translation()
        monomers2, translations2 = cy2._potential_translation()
        idx1, _ = cy1._detect_mirror(
            monomers1[0], translations1[0] * cy1._primitive.cell[2, 2]
        )
        idx2, _ = cy2._detect_mirror(
            monomers2[0], translations2[0] * cy2._primitive.cell[2, 2]
        )
        assert idx1 == True
        assert idx2 == False

    def test_the_whole_function_st1(self, shared_datadir):
        st_name = shared_datadir / "st1"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st, tolerance=1e-2)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "(C12|T3(1.5))" and str(mon[0].symbols) == "C4"

    def test_the_whole_function_st2(self, shared_datadir):
        st_name = shared_datadir / "st7"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st, tolerance=1e-2)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "(C24|T2(1.5))" and str(mon[0].symbols) == "C6"

    def test_the_whole_function_st3(self, shared_datadir):
        st_name = shared_datadir / "9-9-AM"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st, tolerance=1e-2)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "(C18|T2(1.606))" and str(mon[0].symbols) == "Mo9S18"

    def test_the_whole_function_st4(self, shared_datadir):
        st_name = shared_datadir / "24-0-ZZ"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st, tolerance=1e-2)
        cy, mon = cyclic.get_cyclic_group()
        assert (
            cy[0] == "(C48|T2(2.74))"
            and cy[1] == "T'(2.74)"
            and str(mon[0].symbols) == "Mo24S48"
        )


class TestAxialPointGroupAnalyzer:
    def test_axial_pg_st1(self, shared_datadir):
        st_name = shared_datadir / "m1"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "D6h"

    def test_axial_pg_st2(self, shared_datadir):
        st_name = shared_datadir / "m2"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "D6d"

    def test_axial_pg_st3(self, shared_datadir):
        st_name = shared_datadir / "m4"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "D6"

    def test_axial_pg_st4(self, shared_datadir):
        st_name = shared_datadir / "9-9-AM"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "S18"

    def test_axial_pg_st6(self, shared_datadir):
        st_name = shared_datadir / "C4h"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C4h"

    def test_axial_pg_st7(self, shared_datadir):
        st_name = shared_datadir / "C4v"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C4v"

    def test_axial_pg_st8(self, shared_datadir):
        st_name = shared_datadir / "C4"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C4"

    def test_axial_pg_st9(self, shared_datadir):
        st_name = shared_datadir / "non-sym"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C1"

    def test_axial_pg_st10(self, shared_datadir):
        st_name = shared_datadir / "24-0-ZZ"
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(st)
        pg = obj.get_pointgroup()
        assert str(pg) == "C24v"


class TestCyclicGroupAnalyzerMethods:
    """Test individual methods of CyclicGroupAnalyzer."""

    def test_get_cyclic_group_and_op(self, shared_datadir):
        """get_cyclic_group_and_op returns groups, monomers, and operations."""
        st = read_vasp(shared_datadir / "st7")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        groups, monomers, ops = cy.get_cyclic_group_and_op()
        assert len(groups) == len(monomers) == len(ops)
        assert len(groups) > 0

    def test_get_generators_screw_axis(self, shared_datadir):
        """Generator for a screw-axis group should be a 4x4 affine matrix."""
        st = read_vasp(shared_datadir / "9-9-AM")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        gen = cy.get_generators()
        assert gen.shape == (4, 4)
        # Should contain a rotation (not identity)
        assert not np.allclose(gen[:3, :3], np.eye(3))

    def test_get_generators_pure_translation(self, shared_datadir):
        """Generator for a T-only group should be 4x4 affine matrix."""
        st = read_vasp(shared_datadir / "C4")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        gen = cy.get_generators()
        assert gen.shape == (4, 4)

    def test_center_of_nanotube(self, shared_datadir):
        """_center_of_nanotube returns a valid centered structure."""
        st = read_vasp(shared_datadir / "12-12-AM")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        centered = cy._center_of_nanotube(st)
        assert len(centered) == len(st)
        assert centered.cell is not None

    def test_check_if_along_OZ_true(self, shared_datadir):
        st = read_vasp(shared_datadir / "st1")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        assert cy._check_if_along_OZ(st) is True

    def test_check_if_along_OZ_false(self):
        """A cell with off-diagonal z-components should fail the OZ check."""
        from ase import Atoms

        bad_atom = Atoms(
            "C2",
            positions=[[0, 0, 0], [1, 1, 1]],
            cell=[[10, 0, 0], [0, 10, 0], [1, 0, 5]],
        )
        # Need an existing CyclicGroupAnalyzer instance to call the method
        # Use a valid structure to create one, then test with bad cell
        bad_atom2 = Atoms(
            "C2",
            positions=[[0, 0, 0], [1, 1, 1]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 5]],
        )
        cy = CyclicGroupAnalyzer(bad_atom2, tolerance=1e-2)
        assert cy._check_if_along_OZ(bad_atom) is False

    def test_detect_rotation_ind1_returns_false(self, shared_datadir):
        """When ind==1, _detect_rotation should return False immediately."""
        st = read_vasp(shared_datadir / "st1")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        monomers, translations = cy._potential_translation()
        found, Q, ops = cy._detect_rotation(
            monomers[0], translations[0] * cy._primitive.cell[2, 2], ind=1
        )
        assert found is False
        assert Q == 1
        assert ops is None

    def test_find_primitive_already_primitive(self, shared_datadir):
        """When the structure is already primitive, _find_primitive returns it unchanged."""
        st = read_vasp(shared_datadir / "C4")
        cy = CyclicGroupAnalyzer(st, tolerance=1e-2)
        assert len(cy._primitive) == len(cy._atom)
        assert cy.supercell_mutiple == 1


class TestLineGroupAnalyzerMethods:
    """Test individual methods of LineGroupAnalyzer."""

    def test_get_symmetry_operations_D6h(self, shared_datadir):
        """D6h should have 24 symmetry operations."""
        st = read_vasp(shared_datadir / "m1")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        ops = obj.get_symmetry_operations()
        assert len(ops) == 24
        assert all(isinstance(op, SymmOp) for op in ops)

    def test_get_symmetry_operations_C1(self, shared_datadir):
        """C1 should have only the identity."""
        st = read_vasp(shared_datadir / "non-sym")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        ops = obj.get_symmetry_operations()
        assert len(ops) == 1

    def test_get_generators_D6h(self, shared_datadir):
        st = read_vasp(shared_datadir / "m1")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        gens = obj.get_generators()
        assert len(gens) > 0
        for g in gens:
            assert g.shape == (4, 4)

    def test_get_generators_excludes_identity(self, shared_datadir):
        st = read_vasp(shared_datadir / "m1")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        gens = obj.get_generators()
        for g in gens:
            assert not np.allclose(g, np.eye(4))

    def test_inertia_tensor_symmetric(self, shared_datadir):
        st = read_vasp(shared_datadir / "m1")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        I = obj._inertia_tensor()
        assert I.shape == (3, 3)
        assert np.allclose(I, I.T, atol=1e-10)

    def test_center_of_mass_periodic(self, shared_datadir):
        st = read_vasp(shared_datadir / "12-12-AM")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        com = obj._get_center_of_mass_periodic(st)
        assert com.shape == (3,)
        assert np.all(com >= 0) and np.all(com <= 1)

    def test_find_axis_center(self, shared_datadir):
        st = read_vasp(shared_datadir / "12-12-AM")
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        centered = obj._find_axis_center_of_nanotube(st)
        assert len(centered) == len(st)
        # After centering, the mean xy position should be near cell center
        scaled = centered.get_scaled_positions()
        mean_xy = scaled[:, :2].mean(axis=0)
        assert np.allclose(mean_xy, [0.5, 0.5], atol=0.05)

    def test_accepts_atoms_object(self, shared_datadir):
        """LineGroupAnalyzer should accept ASE Atoms directly."""
        st = read_vasp(shared_datadir / "12-12-AM")
        obj = LineGroupAnalyzer(st, tolerance=0.01)
        pg = obj.get_pointgroup()
        assert str(pg) != ""


class TestGetSymcell:
    """Test get_symcell function."""

    def test_symcell_reduces_atoms(self, shared_datadir):
        """Symmetry cell should have fewer atoms than the full monomer."""
        st = read_vasp(shared_datadir / "m1")
        symcell = get_symcell(st)
        assert len(symcell) < len(st)

    def test_symcell_nonempty(self, shared_datadir):
        st = read_vasp(shared_datadir / "m1")
        symcell = get_symcell(st)
        assert len(symcell) >= 1
