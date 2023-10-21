# Copyright 2023 The PULGON Project Developers
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

import os
from pathlib import Path
from pdb import set_trace

import pretty_errors
import pytest
import pytest_datadir
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer

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
        assert Q == 12

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
        assert translations[0] == 0.5

    def test_rotational_tolerance(self, shared_datadir):
        st_name = shared_datadir / "st1"
        st = read_vasp(st_name)
        cy1 = CyclicGroupAnalyzer(st, tolerance=1e-2)
        cy2 = CyclicGroupAnalyzer(st, tolerance=1e-3)
        monomers1, translations1 = cy1._potential_translation()
        monomers2, translations2 = cy2._potential_translation()
        idx1, Q1, _ = cy1._detect_rotation(
            monomers1[0], translations1[0] * cy1._primitive.cell[2, 2], ind=3
        )
        idx2, Q2, _ = cy2._detect_rotation(
            monomers2[0], translations2[0] * cy2._primitive.cell[2, 2], ind=3
        )
        assert idx1 == True and Q1 == 12
        assert idx2 == False and Q2 == 1

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
        assert cy[0] == "T12(1.498)" and str(mon[0].symbols) == "C4"

    def test_the_whole_function_st2(self, shared_datadir):
        st_name = shared_datadir / "st7"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "T'(1.5)" and str(mon[0].symbols) == "C6"

    def test_the_whole_function_st3(self, shared_datadir):
        st_name = shared_datadir / "9-9-AM"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st, tolerance=0.01)
        cy, mon = cyclic.get_cyclic_group()
        assert cy[0] == "T18(1.614)" and str(mon[0].symbols) == "Mo9S18"

    def test_the_whole_function_st4(self, shared_datadir):
        st_name = shared_datadir / "24-0-ZZ"
        st = read_vasp(st_name)
        cyclic = CyclicGroupAnalyzer(st, tolerance=0.01)
        cy, mon = cyclic.get_cyclic_group()
        assert (
            cy[0] == "T48(2.74)"
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
        assert str(pg) == "S2"

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
