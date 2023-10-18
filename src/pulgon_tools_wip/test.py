import argparse
import json
import pickle
import typing
from ast import literal_eval
from pdb import set_trace

import numpy as np
from ase.io.vasp import read_vasp, write_vasp
from utils import get_symcell

from pulgon_tools_wip.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer


def main():
    poscar = read_vasp("POSCAR")
    cyclic = CyclicGroupAnalyzer(poscar, corner=True, symprec=0.01)
    cy, monomer = cyclic.get_cyclic_group()
    symcell = get_symcell(monomer[0])

    write_vasp("monomer.vasp", monomer[0])
    write_vasp("symcell.vasp", symcell)


if __name__ == "__main__":
    main()
