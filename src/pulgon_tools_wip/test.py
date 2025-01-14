import re

import numpy as np
import pytest
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from numpy.linalg.linalg import eigvals
from sympy.printing.octave import print_octave_code

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.generate_structures import (
    Cn,
    S2n,
    U,
    U_d,
    dimino,
    generate_line_group_structure,
    sigmaH,
    sigmaV,
)
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
from pulgon_tools_wip.utils import get_symbols_from_ops


def lingroupfamily(poscar):
    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    trans_sym = cyclic.cyclic_group[0]
    obj = LineGroupAnalyzer(poscar)
    rota_sym = obj.sch_symbol

    print(trans_sym)
    print(rota_sym)

    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    print("family:", family)


def symop_symbol(poscar):
    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    obj = LineGroupAnalyzer(poscar)

    sch, _, op_trans = cyclic.get_cyclic_group_and_op()
    op_rotas = obj.get_generators()

    res = get_symbols_from_ops(op_rotas)
    print(res)


if __name__ == "__main__":
    # poscar = read_vasp("../../test/data/9-9-AM")
    # poscar = read_vasp("../../test/data/12-12-AM")
    # poscar = read_vasp("../../test/data/24-0-ZZ")
    # poscar = read_vasp("../../test/data/C4h")
    # poscar = read_vasp("../../test/data/C4v")
    poscar = read_vasp("../../test/data/C6u")
    # poscar = read_vasp("../../test/data/C4")
    lingroupfamily(poscar)
    symop_symbol(poscar)
