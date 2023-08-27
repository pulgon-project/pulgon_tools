import argparse
import copy
import itertools
from pdb import set_trace

import numpy as np
import pretty_errors
import spglib
from ase import Atoms
from ase.data import atomic_masses
from ase.geometry import find_mic
from ase.io import read
from ase.io.vasp import read_vasp
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
from pymatgen.util.coord import find_in_coord_list
from spglib import get_symmetry, get_symmetry_dataset

from pulgon_tools_wip.utils import refine_cell


class CyclicGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(self, atom, spmprec=0.01, angle_tolerance=5):
        self._symprec = spmprec
        self._angle_tol = angle_tolerance
        self._atom = atom

        self._analyze()

    def _analyze(self):
        primitive = self._find_primitive()
        # pure_trans = primitive.cell[2, 2]
        # potential_trans = self._potential_translation()
        # symmops = self.get_symmetry_operations()
        # tmp = self.is_valid_op(symmops[1])

        monomer_ind = self._detect_monomer()
        monomer = primitive[monomer_ind]

        diff_ind = np.setdiff1d(range(len(primitive)), monomer_ind)
        diff_st = primitive[diff_ind]

        translations = self.get_translations(diff_st, monomer)
        set_trace()

    def get_translations(self, atoms, monomer_atoms):
        """
        获取给定monomer在结构中所有可能的广义平移位置。

        :param structure: ase的Atoms对象
        :param monomer_atoms: 结构中定义的monomer的atoms对象
        :return: 一个列表，包含monomer所有可能的广义平移位置
        """
        translations = []

        # 对每一个原子，找出与monomer中第一个原子的最小映像规定的位移
        for atom in atoms:
            # set_trace()
            displacement, distance = find_mic(
                [atom.position - monomer_atoms[0].position], atoms.cell
            )
            if distance < 0.01:  # 使用小于0.01作为两原子相同的判据
                translations.append(displacement)

        # detect rotation
        rotation = self._detect_rotation()
        # detect mirror
        mirror = self._detect_mirror()

        return {
            "translation": translations,
            "rotation": rotation,
            "mirror": mirror,
        }

    def _detect_rotation(self):
        pass

    def _detect_mirror(self):
        pass

    def is_valid_op(self, symmop) -> bool:
        """Check if a particular symmetry operation is a valid symmetry operation for,
         i.e., the operation maps all atoms to another equivalent atom.

        Args:
            symmop (SymmOp): Symmetry operation to test.

        Returns:
            (bool): Whether SymmOp is valid.
        """
        coords = self._atom.get_scaled_positions()
        for ii, site in enumerate(coords):
            coord = symmop.operate(site)
            ref_coord, _ = refine_cell(coord, self._atom.numbers[ii])

            set_trace()
            ind = find_in_coord_list(coords, ref_coord, self._symprec)
            if not (
                len(ind) == 1 and self._atom.number[ind[0]] == site.species
            ):
                return False
        return True

    def _detect_monomer(self):
        monomer_ind = np.array([0, 1, 6, 7])

        return monomer_ind

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
