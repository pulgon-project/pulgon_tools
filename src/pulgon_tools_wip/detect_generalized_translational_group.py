import argparse
from pdb import set_trace

import numpy as np
import pretty_errors
from ase.data import atomic_masses
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer


def main():
    poscar = "st1.vasp"
    st = read_vasp(poscar)

    set_trace()


if __name__ == "__main__":
    main()
