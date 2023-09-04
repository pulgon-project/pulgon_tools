import os
from pathlib import Path
from pdb import set_trace

import pretty_errors
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer

MODULE_DIR = Path(__file__).absolute().parent


class TestCyclicGroupAnalyzer:
    def test_find_axis_center_of_nanotube(self):
        st_name = os.path.join(MODULE_DIR, "data", "12-12-AM.vasp")
        st = read_vasp(st_name)
        res = CyclicGroupAnalyzer(st)
        n_st = res._find_axis_center_of_nanotube(st)
        average_coord = (n_st.positions[:, :2] / len(n_st)).sum(axis=0)
        assert (average_coord - [0, 0]).sum() < 0.001

    def test_generate_monomer(self):
        st_name = os.path.join(MODULE_DIR, "data", "9-9-AM.vasp")
        st = read_vasp(st_name)
        cy = CyclicGroupAnalyzer(st)
        monomers, translations = cy._potential_translation()
        assert str(monomers[0].symbols) == "Mo9S18"
        assert translations[0] == 0.5

    def test_rotation(self):
        st_name = os.path.join(MODULE_DIR, "data", "st1.vasp")
        st = read_vasp(st_name)
        cy = CyclicGroupAnalyzer(st)
        monomers, translations = cy._potential_translation()
        idx, Q = cy._detect_rotation(
            monomers[0], translations[0] * cy._primitive.cell[2, 2], 3
        )
        assert idx == True
        assert Q == 12

    def test_rotational_tolerance(self):
        st_name = os.path.join(MODULE_DIR, "data", "st1.vasp")
        st = read_vasp(st_name)
        cy1 = CyclicGroupAnalyzer(st, symprec=1e-4)
        cy2 = CyclicGroupAnalyzer(st, symprec=1e-5)
        monomers1, translations1 = cy1._potential_translation()
        monomers2, translations2 = cy2._potential_translation()
        idx1, Q1 = cy1._detect_rotation(
            monomers1[0], translations1[0] * cy1._primitive.cell[2, 2], ind=3
        )
        idx2, Q2 = cy2._detect_rotation(
            monomers2[0], translations2[0] * cy2._primitive.cell[2, 2], ind=3
        )
        assert idx1 == True and Q1 == 12
        assert idx2 == False and Q2 == 1

    def test_mirror(self):
        st_name = os.path.join(MODULE_DIR, "data", "st7.vasp")
        st = read_vasp(st_name)
        cy = CyclicGroupAnalyzer(st)
        monomers, translations = cy._potential_translation()
        idx = cy._detect_mirror(
            monomers[0], translations[0] * cy._primitive.cell[2, 2]
        )
        assert idx == True

    def test_mirror_tolerance(self):
        st_name = os.path.join(MODULE_DIR, "data", "st7.vasp")
        st = read_vasp(st_name)

        cy1 = CyclicGroupAnalyzer(st, symprec=1e-15)
        cy2 = CyclicGroupAnalyzer(st, symprec=1e-16)
        monomers1, translations1 = cy1._potential_translation()
        monomers2, translations2 = cy2._potential_translation()
        idx1 = cy1._detect_mirror(
            monomers1[0], translations1[0] * cy1._primitive.cell[2, 2]
        )
        idx2 = cy2._detect_mirror(
            monomers2[0], translations2[0] * cy2._primitive.cell[2, 2]
        )
        assert idx1 == True
        assert idx2 == False

    def test_the_whole_function_st1(self):
        st_name = os.path.join(MODULE_DIR, "data", "st1.vasp")
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "T12(1.5)" and str(mon[0].symbols) == "C4"

    def test_the_whole_function_st2(self):
        st_name = os.path.join(MODULE_DIR, "data", "st7.vasp")
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "T'(1.5)" and str(mon[0].symbols) == "C6"

    def test_the_whole_function_st3(self):
        st_name = os.path.join(MODULE_DIR, "data", "9-9-AM.vasp")
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "T2(1.614)" and str(mon[0].symbols) == "Mo9S18"


class TestAxialPointGroupAnalyzer:
    def test_axial_pg_st1(self):
        st_name = os.path.join(MODULE_DIR, "data", "m1")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "D6h"

    def test_axial_pg_st2(self):
        st_name = os.path.join(MODULE_DIR, "data", "m2")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "D6d"

    def test_axial_pg_st3(self):
        st_name = os.path.join(MODULE_DIR, "data", "m4")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "D6"

    def test_axial_pg_st4(self):
        st_name = os.path.join(MODULE_DIR, "data", "9-9-AM.vasp")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "S2"

    def test_axial_pg_st6(self):
        st_name = os.path.join(MODULE_DIR, "data", "C4h.vasp")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C4h"

    def test_axial_pg_st7(self):
        st_name = os.path.join(MODULE_DIR, "data", "C4v.vasp")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C4v"

    def test_axial_pg_st8(self):
        st_name = os.path.join(MODULE_DIR, "data", "C4.vasp")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C4"

    def test_axial_pg_st9(self):
        st_name = os.path.join(MODULE_DIR, "data", "non-sym.vasp")
        st = read_vasp(st_name)
        mol = Molecule(species=st.numbers, coords=st.positions)
        obj = LineGroupAnalyzer(mol)
        pg = obj.get_pointgroup()
        assert str(pg) == "C1"
