from pdb import set_trace

import pytest_datadir
from ase.io.vasp import read_vasp

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.utils import get_perms


def test_get_perms_st1(shared_datadir):
    poscar = read_vasp(shared_datadir / "24-0-ZZ")

    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    cy, mon, sym_cy_ops = cyclic.get_cyclic_group_and_op()
    atom = cyclic._primitive

    obj = LineGroupAnalyzer(poscar)
    sym_pg_ops = obj.get_symmetry_operations()

    perms_table, _ = get_perms(atom, sym_cy_ops[0], sym_pg_ops)
    assert len(perms_table) == 96


def test_get_perms_st2(shared_datadir):
    poscar = read_vasp(shared_datadir / "9-9-AM")
    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    cy, mon, sym_cy_ops = cyclic.get_cyclic_group_and_op()
    atom = cyclic._primitive

    obj = LineGroupAnalyzer(poscar)
    sym_pg_ops = obj.get_symmetry_operations()
    perms_table, _ = get_perms(atom, sym_cy_ops[0], sym_pg_ops)
    assert len(perms_table) == 18
