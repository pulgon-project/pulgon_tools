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
    def __init__(self, atom, spmprec=0.01, angle_tolerance=5, round_symprec=3):
        self._symprec = spmprec
        self._round_symprec = round_symprec
        self._angle_tol = angle_tolerance
        self._zaxis = np.array([0, 0, 1])
        self._atom = atom
        # Todo: pure translations and primitive cell

        self._primitive = atom
        self._pure_trans = self._primitive.cell[2, 2]

        # Todo: find out the x-y center
        self._primitive = self._center_of_xy(self._primitive)

        self._analyze()

    def _analyze(self):
        monomer, potential_trans = self._potential_translation()

        cyclic_group, monomers = self._get_translations(
            monomer, potential_trans
        )

        for ii, cg in enumerate(cyclic_group):
            print("cyclic_group: " + cg + "  " + str(monomers[ii].symbols))

    def _center_of_xy(self, atom):
        n_st = atom.copy()
        vector = atom.get_center_of_mass()
        n_st.positions = n_st.positions - [vector[0], vector[1], 0]
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

            ind = int(np.round(1 / tran, self._round_symprec))
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
                        "T%s(%s)" % (Q, np.round(tran * self._pure_trans, 3))
                    )
                    mono.append(monomer)
                if (
                    ind == 2 and abs(tran - 0.5) < self._symprec
                ):  # only 2 layer in primitive cell
                    # detect mirror
                    coords = self._primitive.positions
                    diff_st_ind = np.array(
                        [
                            find_in_coord_list(coords, coord, self._symprec)
                            for coord in monomer.positions
                        ]
                    )
                    if diff_st_ind.ndim > 1:
                        diff_st_ind = diff_st_ind.T[0]

                    diff_st = self._primitive[
                        np.setdiff1d(range(len(coords)), diff_st_ind)
                    ]

                    mirror = self._detect_mirror(
                        monomer, diff_st, self._pure_trans / 2
                    )
                    if mirror:
                        cyclic_group.append(
                            "T'(%s)" % np.round(self._pure_trans / 2, 3)
                        )
                        mono.append(monomer)
        return cyclic_group, mono

    def _detect_rotation(self, monomer, tran, ind):
        coords = self._primitive.positions

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

            itp1, itp2 = (
                True,
                True,
            )  # record the rotational result from different layer
            for layer in range(1, ind):
                op1 = SymmOp.from_axis_angle_and_translation(
                    self._zaxis,
                    test_ind * layer,
                    translation_vec=(0, 0, tran * layer),
                )
                op2 = SymmOp.from_axis_angle_and_translation(
                    self._zaxis,
                    -test_ind * layer,
                    translation_vec=(0, 0, tran * layer),
                )
                itp3, itp4 = (
                    [],
                    [],
                )  # record the rotational result in current layer
                for site in monomer:
                    coord1 = op1.operate(site.position)
                    coord2 = op2.operate(site.position)

                    tmp1 = find_in_coord_list(coords, coord1, self._symprec)
                    tmp2 = find_in_coord_list(coords, coord2, self._symprec)
                    itp3.append(
                        len(tmp1) == 1
                        and self._primitive.numbers[tmp1[0]] == site.number
                    )
                    itp4.append(
                        len(tmp2) == 1
                        and self._primitive.numbers[tmp2[0]] == site.number
                    )
                itp1 = itp1 and np.array(itp3).all()
                itp2 = itp2 and np.array(itp4).all()
                if not (itp1 or itp2):
                    break

            # set_trace()
            if itp1 or itp2:
                Q = int(360 / test_ind)
                return True, Q
        return False, 1

    def _detect_mirror(self, monomer, diff_st, tran):
        for itp1, itp2 in itertools.combinations_with_replacement(
            range(len(monomer)), 2
        ):
            s1, s2 = monomer[itp1], diff_st[itp2]

            if (
                s1.number == s2.number
                and (s1.position[2] + tran - s2.position[2]) < self._symprec
            ):
                normal = s1.position - s2.position
                normal[2] = 0
                op = SymmOp.reflection(normal)
                # set_trace()

                itp = []
                for site in monomer:
                    coord = op.operate(site.position) + np.array([0, 0, tran])
                    tmp = find_in_coord_list(
                        diff_st.positions, coord, self._symprec
                    )
                    itp.append(
                        len(tmp) == 1
                        and diff_st.numbers[tmp[0]] == site.number
                    )
                if np.array(itp).all():
                    return True
        return False

    def _potential_translation(self):
        """generate the potential monomer and the scaled translational distance in z axis

        Returns:

        """

        z = self._primitive.get_scaled_positions()[:, 2]
        z_uniq, counts = np.unique(z, return_counts=True)
        potential_trans = np.append((z_uniq - z_uniq[0])[1:], 1)
        monomer_ind = [np.where(z == tmp)[0] for tmp in z_uniq]

        monomer_ind_sum = []
        tmp1 = np.array([])
        for tmp in monomer_ind:
            tmp1 = np.sort(np.append(tmp1, tmp)).astype(np.int32)
            monomer_ind_sum.append(tmp1)

        translation, monomer = [], []
        for ii in range(len(z_uniq)):
            monomer_num = counts[: ii + 1].sum()
            # check the atomic number and layer number of potential monomer
            # check the translational distance whether correspond to the layer numbers
            if (
                len(self._primitive) % monomer_num == 0
                and len(z_uniq) % (ii + 1) == 0
                and abs(len(z_uniq) / (ii + 1) - 1 / potential_trans[ii])
                < self._symprec
            ):
                # set_trace()

                if len(self._primitive) == monomer_num:
                    # if the monomer is the whole structure
                    monomer.append(self._primitive)
                    translation.append(1)
                else:
                    monomer.append(self._primitive[monomer_ind_sum[ii]])
                    translation.append(potential_trans[ii])
        # set_trace()
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
    poscar = "st13.vasp"
    atom = read_vasp(poscar)
    # dataset = get_symmetry_dataset(st, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)
    # res1 = spglib.find_primitive(st)
    res = CyclicGroupAnalyzer(atom)

    # set_trace()


if __name__ == "__main__":
    main()
