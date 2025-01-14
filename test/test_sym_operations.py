import pytest_datadir
from ase.io.vasp import read_vasp
from ipdb import set_trace

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
from pulgon_tools_wip.utils import get_perms, get_symbols_from_ops


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


def test_lingroupfamily(shared_datadir):
    poscar = read_vasp(shared_datadir / "9-9-AM")

    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    trans_sym = cyclic.cyclic_group[0]
    obj = LineGroupAnalyzer(poscar)
    rota_sym = obj.sch_symbol
    # family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    family = get_family_Num_from_sym_symbol("T", "C1")

    assert family == 4


def test_symop_symbol(shared_datadir):
    poscar = read_vasp(shared_datadir / "9-9-AM")

    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    obj = LineGroupAnalyzer(poscar)

    sch, _, op_trans = cyclic.get_cyclic_group_and_op()
    op_rotas = obj.get_generators()
    symbols = get_symbols_from_ops(op_rotas)
    assert symbols[0] == "C9" and symbols[1] == "S18"
