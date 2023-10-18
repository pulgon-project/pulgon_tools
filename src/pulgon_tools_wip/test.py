import argparse
import json
import pickle
import typing
from ast import literal_eval
from pdb import set_trace

import numpy as np
from ase.io.vasp import read_vasp, write_vasp

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer


def get_symcell(monomer):
    apg = LineGroupAnalyzer(monomer)
    equ = list(apg.get_equivalent_atoms()["eq_sets"])
    # sym = apg.get_symmetry_operations()
    # write_vasp("monomer.vasp", monomer)
    # write_vasp("symcell.vasp", monomer[equ])
    return monomer[equ]


def main():
    poscar = read_vasp("POSCAR")
    cyclic = CyclicGroupAnalyzer(poscar, corner=True, symprec=0.01)
    cy, mon = cyclic.get_cyclic_group()

    get_symcell(mon[0])


if __name__ == "__main__":
    main()
