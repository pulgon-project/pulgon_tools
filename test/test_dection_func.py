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


class TestLineGroupAnalyzer:
    pass


class TestCyclicGroupAnalyzer:
    pass


def test_axial_pg_m1():
    st_name = os.path.join(MODULE_DIR, "data", "m1")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_axial_pg_m2():
    st_name = os.path.join(MODULE_DIR, "data", "m2")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6d"


def test_axial_pg_m3():
    st_name = os.path.join(MODULE_DIR, "data", "m3")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_axial_pg_m4():
    st_name = os.path.join(MODULE_DIR, "data", "m4")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6"


def test_axial_pg_m5():
    st_name = os.path.join(MODULE_DIR, "data", "m5")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_axial_pg_m6():
    st_name = os.path.join(MODULE_DIR, "data", "m6")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6d"


def test_axial_pg_m7():
    st_name = os.path.join(MODULE_DIR, "data", "m7")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_axial_pg_m8():
    st_name = os.path.join(MODULE_DIR, "data", "m8")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D4h"


def test_cyclic_group_st1():
    st_name = os.path.join(MODULE_DIR, "data", "st1")
    st = read_vasp(st_name)
    cyclic = CyclicGroupAnalyzer(st)
    cy, mon = cyclic.get_cyclic_group()
    assert cy[0] == "T12(1.5)" and str(mon[0].symbols) == "C4"


def test_cyclic_group_st2():
    st_name = os.path.join(MODULE_DIR, "data", "st4")
    st = read_vasp(st_name)
    cyclic = CyclicGroupAnalyzer(st)
    cy, mon = cyclic.get_cyclic_group()
    assert (
        cy[0] == "T12(4.0)"
        and str(mon[0].symbols) == "C12"
        and cy[1] == "T'(4.0)"
        and str(mon[1].symbols) == "C12"
    )


def test_cyclic_group_st3():
    st_name = os.path.join(MODULE_DIR, "data", "st5")
    st = read_vasp(st_name)
    cyclic = CyclicGroupAnalyzer(st)
    cy, mon = cyclic.get_cyclic_group()
    assert cy[0] == "T12(4.0)" and str(mon[0].symbols) == "C12"
