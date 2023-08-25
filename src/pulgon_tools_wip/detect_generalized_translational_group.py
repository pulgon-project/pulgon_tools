import argparse
import copy
import itertools
from pdb import set_trace

import numpy as np
import pretty_errors
import spglib
from ase import Atoms
from ase.data import atomic_masses
from ase.io import read
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
from spglib import get_symmetry, get_symmetry_dataset


class CyclicGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(self, atom, spmprec=0.01, angle_tolerance=5):
        self._symprec = spmprec
        self._angle_tol = angle_tolerance
        self._atom = atom

        self._analyze()

    def _analyze(self):
        primitive = self._find_primitive()
        pure_trans = primitive.cell[2, 2]

        potential_trans = self._potential_translation()
        symmops = self.get_symmetry_operations()

        set_trace()
        monomer = self._detect_monomer()

    def _detect_monomer(self):
        monomer = []

        return monomer

    def _potential_translation(self):
        translation = []

        return translation

    def _find_primitive(self):
        lattice, scaled_positions, numbers = spglib.find_primitive(
            self._atom, symprec=self._symprec
        )
        primitive = Atoms(
            cell=lattice, scaled_positions=scaled_positions, numbers=numbers
        )
        return primitive

    def get_symmetry_operations(self, cartesian=False):
        """Return symmetry operations as a list of SymmOp objects. By default returns
        fractional coord symmops. But Cartesian can be returned too.

        Returns:
            ([SymmOp]): List of symmetry operations.
        """
        sym = spglib.get_symmetry(
            self._atom, symprec=self._symprec, angle_tolerance=self._angle_tol
        )
        rotation, translation = sym["rotations"], sym["translations"]
        symmops = []
        # mat = self._structure.lattice.matrix.T
        mat = np.array(self._atom.cell)
        invmat = np.linalg.inv(mat)
        for rot, trans in zip(rotation, translation):
            if cartesian:
                rot = np.dot(mat, np.dot(rot, invmat))
                trans = np.dot(trans, self._structure.lattice.matrix)
            op = SymmOp.from_rotation_and_translation(rot, trans)
            symmops.append(op)
        return symmops


def main():
    poscar = "st1.vasp"
    atom = read_vasp(poscar)

    # dataset = get_symmetry_dataset(st, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)

    # res1 = spglib.find_primitive(st)
    res = CyclicGroupAnalyzer(atom)

    set_trace()


if __name__ == "__main__":
    main()
