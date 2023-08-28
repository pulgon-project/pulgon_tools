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


class CyclicGroupAnalyzer:
    def __init__(self, atom, spmprec=0.01, angle_tolerance=5):
        self._symprec = spmprec
        self._angle_tol = angle_tolerance
        self._zaxis = np.array([0, 0, 1])
        self._atom = atom
        self._primitive = self._find_primitive()
        self._pure_trans = self._primitive.cell[2, 2]
        self._primitive = self._center_of_atom(self._primitive)

        self._analyze()

    def _analyze(self):
        # symmops = self.get_symmetry_operations()
        # tmp = self.is_valid_op(symmops[1])
        # diff_ind = np.setdiff1d(range(len(primitive)), monomer_ind)
        # diff_st = primitive[diff_ind]

        monomer, potential_trans = self._potential_translation()

        cyclic_group, _ = self._get_translations(monomer, potential_trans)
        set_trace()

    def _center_of_atom(self, atom):
        n_st = atom.copy()
        vector = atom.get_center_of_mass()
        n_st.positions = n_st.positions - vector

        return n_st

    def _get_translations(self, monomer_atoms, potential_tans):
        """

        Args:
            atoms:
            monomer_atoms:
            trans:

        Returns:

        """

        cyclic_group, mono = [], []
        for ii, monomer in enumerate(monomer_atoms):
            tran = potential_tans[ii]

            ind = int(1 / tran)
            if ind - 1 / tran > self._symprec:
                print("selecting wrong translation vector")
                continue

            if len(monomer) == len(self._primitive):
                cyclic_group.append("T")
                mono.append(monomer)
            else:
                # detect rotation
                rotation, Q = self._detect_rotation(
                    monomer, tran * self._pure_trans, ind
                )

                if rotation:
                    cyclic_group.append(
                        "T%s(%s)" % (Q, tran * self._pure_trans)
                    )
                    mono.append(monomer)
                if ind == 2:
                    # detect mirror
                    mirror = self._detect_mirror(
                        monomer, tran * self._pure_trans
                    )
                    if mirror:
                        cyclic_group.append("T'(%s)" % tran * self._pure_trans)
                        mono.append(monomer)
        return cyclic_group, mono

    def _detect_rotation(self, monomer, tran, ind):

        # detect the monomer's rotational symmetry for specifying therotation
        mol = Molecule(species=monomer.numbers, coords=monomer.positions)
        monomer_rot_ind = PointGroupAnalyzer(mol)._check_rot_sym(self._zaxis)

        # possible rotational angle in cyclic group
        ind1 = (
            np.array(
                [
                    360 * ii / monomer_rot_ind
                    for ii in range(1, monomer_rot_ind + 1)
                ]
            )
            / ind
        )

        for test_ind in ind1:
            op1 = SymmOp.from_axis_angle_and_translation(
                self._zaxis, test_ind, translation_vec=(0, 0, tran)
            )
            op2 = SymmOp.from_axis_angle_and_translation(
                self._zaxis, -test_ind, translation_vec=(0, 0, tran)
            )

            coords = self._primitive.positions

            itp1, itp2 = [], []
            for site in monomer:
                coord1 = op1.operate(site.position)
                coord2 = op2.operate(site.position)

                tmp1 = find_in_coord_list(coords, coord1, self._symprec)
                tmp2 = find_in_coord_list(coords, coord2, self._symprec)
                itp1.append(
                    len(tmp1) == 1
                    and self._primitive.numbers[tmp1[0]] == site.number
                )
                itp2.append(
                    len(tmp2) == 1
                    and self._primitive.numbers[tmp2[0]] == site.number
                )
            itp1 = np.array(itp1)
            itp2 = np.array(itp2)
            set_trace()

        if not (
            len(itp1) == 1
            and self.centered_mol[itp1[0]].species == site.species
        ):
            return False, 1

        return True, Q

    def _detect_mirror(self, monomer, tran):
        pass

    def _potential_translation(self):
        """generate the potential monomer and the scaled translational distance in z axis

        Returns:

        """

        z = self._primitive.get_scaled_positions()[:, 2]
        z_uniq, counts = np.unique(z, return_counts=True)
        potential_trans = (z_uniq - z_uniq[0])[1:]
        monomer_ind = [np.where(z == tmp)[0] for tmp in z_uniq]

        translation, monomer = [], []
        for ii in range(len(z_uniq)):
            monomer_num = counts[: ii + 1].sum()
            # check the atomic number and layer number of potential monomer
            # can't identify all the situations but exception can be handle in next step
            if (
                len(self._primitive) % monomer_num == 0
                and len(z_uniq) % (ii + 1) == 0
            ):

                if len(self._primitive) == monomer_num:
                    # if the monomer is the whole structure
                    monomer.append(self._primitive)
                    translation.append(1)
                else:
                    monomer.append(self._primitive[monomer_ind[ii]])
                    translation.append(potential_trans[ii])

        return monomer, translation

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
