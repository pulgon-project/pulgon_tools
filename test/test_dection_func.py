import argparse
import os
from pathlib import Path
from pdb import set_trace

import numpy as np
import pretty_errors
from ase.data import atomic_masses
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer

MODULE_DIR = Path(__file__).absolute().parent


def test_m1():
    st_name = os.path.join(MODULE_DIR, "data", "m1")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_m2():
    st_name = os.path.join(MODULE_DIR, "data", "m2")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6d"


def test_m3():
    st_name = os.path.join(MODULE_DIR, "data", "m3")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_m4():
    st_name = os.path.join(MODULE_DIR, "data", "m4")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6"


def test_m5():
    st_name = os.path.join(MODULE_DIR, "data", "m5")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_m6():
    st_name = os.path.join(MODULE_DIR, "data", "m6")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6d"


def test_m7():
    st_name = os.path.join(MODULE_DIR, "data", "m7")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D6h"


def test_m7():
    st_name = os.path.join(MODULE_DIR, "data", "m8")
    st = read_vasp(st_name)
    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    pg = obj.get_pointgroup()
    assert str(pg) == "D4h"
