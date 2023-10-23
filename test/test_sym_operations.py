from pdb import set_trace

import pytest_datadir
from ase.io.vasp import read_vasp

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.utils import get_perms


def test_sym_operations(shared_datadir):
    poscar = read_vasp(shared_datadir / "st7")

    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    cy, mon, sym_cy_ops = cyclic.get_cyclic_group_and_op()
    atom = cyclic._primitive

    atom.center()
    coords1 = atom.positions

    obj = LineGroupAnalyzer(poscar)
    sym_pg_ops = obj.get_symmetry_operations()

    op = sym_pg_ops[0]
    # coords = obj.mol.cart_coords
    coords = obj.centered_mol.cart_coords
    for coord in coords1:
        tmp = op.operate(coord)

        set_trace()


def test_get_perms():
    pass
