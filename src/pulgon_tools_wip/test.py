import argparse
import json
import logging
import pickle
import typing
from ast import literal_eval

import numpy as np
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from pymatgen.core.operations import SymmOp

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.generate_structures import Cn, sigmaH
from pulgon_tools_wip.Irreps_tables import line_group_4
from pulgon_tools_wip.utils import (
    affine_matrix_op,
    dimino_affine_matrix_and_character,
    dimino_affine_matrix_and_subsquent,
)


def get_character(qpoints, Zperiod_a, nrot):
    qpoints = qpoints / Zperiod_a
    Dataset_q = []
    for qz in qpoints:
        Dataset_q.append(line_group_4(Zperiod_a, nrot, qz))
    return Dataset_q


def main():
    poscar = read_vasp("POSCAR")
    cyclic = CyclicGroupAnalyzer(poscar, tolerance=1e-2)
    cy, mon, sym_cy_ops = cyclic.get_cyclic_group_and_op()
    atom = cyclic._primitive

    obj = LineGroupAnalyzer(atom, tolerance=1e-2)
    sym_pg_ops = obj.get_symmetry_operations()
    nrot = obj.get_rotational_symmetry_number()

    # matrices, sym_operations = get_matrices(atom, sym_cy_ops[0], sym_pg_ops)

    NQS = 51
    qpoints = np.linspace(0, np.pi, num=NQS, endpoint=False)
    Zperiod_a = cyclic._pure_trans
    Dataset_q = get_character(qpoints, Zperiod_a, nrot)

    sym = []
    pg1 = [Cn(9), sigmaH()]
    for pg in pg1:
        tmp = SymmOp.from_rotation_and_translation(pg, [0, 0, 0])
        sym.append(tmp.affine_matrix)
    tran = SymmOp.from_rotation_and_translation(Cn(18), [0, 0, 1 / 2])
    sym.append(tran.affine_matrix)

    ops, subs = dimino_affine_matrix_and_subsquent(sym)
    for ii, Dataset in enumerate(Dataset_q[1:]):
        charas = Dataset.character_table
        character = []
        for chara in charas:
            # ops, ops_chara = dimino_affine_matrix_and_character(sym, chara[0])
            if len(ops) != len(subs):
                logging.ERROR("len(ops) != len(subs)")

        set_trace()


if __name__ == "__main__":
    main()
