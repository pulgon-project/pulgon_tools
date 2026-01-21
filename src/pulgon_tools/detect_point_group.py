# Copyright 2023 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import logging
from typing import Union

import ase
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.vasp import write_vasp
from ipdb import set_trace
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from pulgon_tools.utils import (
    brute_force_generate_group,
    find_axis_center_of_nanotube,
)


class LineGroupAnalyzer(PointGroupAnalyzer):
    """A class to analyze the axial point group of a molecule (based on pymatgen:PointGroupAnalyzer)

    The general outline of the algorithm is as follows:

    1. Specify z axis as the rotation axis, detect the rotational symmetry.
    2. If the rotational symmetry about z-axis exist, detect U (a two-fold horizontal axis).
       - If U exist, it's a dihedral group (Dnh, Dnd).
       - If U does not exist, the group is not dihedral, leaving Cnh, Cnv and S2n as candidates.
    3. If the rotational symmetry about z-axis does not exist, only possible point groups are C1, Cs and Ci.
    """

    def __init__(
        self,
        mol: Union[Molecule, Atoms],
        tolerance: float = 0.01,
    ):
        """The default settings are usually sufficient. (Totally the same with PointGroupAnalyzer)

        Args:
            mol (Molecule): Molecule to determine point group.
            tolerance (float): Distance tolerance to consider sites as
                symmetrically equivalent. Defaults to 0.3 Angstrom.
            matrix_tolerance (float): Tolerance used to generate the full set of
                symmetry operations of the point group.
        """
        logging.debug("--------------------start detecting axial point group")

        if type(mol) == Atoms:
            # mol = self._find_axis_center_of_nanotube(mol)
            mol = find_axis_center_of_nanotube(mol)
            mol = Molecule(species=mol.numbers, coords=mol.positions)

        self.mol = mol
        # self.centered_mol = mol
        self.centered_mol = mol.get_centered_molecule()

        self.tol = tolerance
        self.mat_tol = tolerance
        self._zaxis = np.array([0, 0, 1])

        self._analyze()
        # if self.sch_symbol in ["C1v", "C1h"]:
        #     self.sch_symbol = "Cs"

    def _analyze(self):
        """Rewrite the _analyze method, calculate the axial point group elements."""
        inertia_tensor = self._inertia_tensor()
        _, eigvecs = np.linalg.eigh(inertia_tensor)
        self.principal_axes = eigvecs.T  # only be used in _proc_no_rot_sym

        self.rot_sym = []
        self.symmops = [SymmOp(np.eye(4))]

        print("Start detecting rotational number")
        self._check_rot_sym(self._zaxis)
        # if len(self.rot_sym) > 0 and self.rot_sym[0][1]!=1:     # modify the case when i==1
        if len(self.rot_sym) > 0:  # modify the case when i==1
            logging.debug(
                "The rot_num along zaxis is: %d" % self.rot_sym[0][1]
            )
            logging.debug("Start detecting U")

            self._check_perpendicular_r2_axis(self._zaxis)
            if len(self.rot_sym) >= 2:
                logging.debug("U exist, start detecting dihedral group")

                self._proc_dihedral()

            elif len(self.rot_sym) == 1:
                logging.debug(
                    "U does not exist, leaving Cnh, Cnv and S2n as candidates"
                )
                self._proc_cyclic()
        else:
            logging.debug("The rot symmetry along zaxis does not exist")
            logging.debug(
                "leaving Ci, C1h and C1v as candidates, start detecting U, v, h"
            )
            self._proc_no_rot_sym()

    def _inertia_tensor(self) -> np.ndarray:
        """

        Returns: inertia_tensor of the molecular

        """

        weights = np.array([site.species.weight for site in self.centered_mol])
        coords = self.centered_mol.cart_coords
        total_inertia = np.sum(weights * np.sum(coords**2, axis=1))

        # nondiagonal terms + diagonal terms
        inertia_tensor = (
            (np.ones((3, 3)) - np.eye(3))
            * (
                np.swapaxes(np.tile(weights, (3, 3, 1)), 0, 2)
                * coords[:, np.tile([[0], [1], [2]], (1, 3))]
                * coords[:, np.tile([0, 1, 2], (3, 1))]
            ).sum(axis=0)
            + (
                ((coords**2).sum(axis=1) * weights).sum()
                - (
                    (coords**2)
                    * np.tile(weights.reshape(weights.shape[0], 1), 3)
                ).sum(axis=0)
            )
            * np.eye(3)
        ) / total_inertia
        return inertia_tensor

    def _get_center_of_mass_periodic(self, atom):
        cell_max = [1, 1, 1]
        tmp = atom.get_scaled_positions() / cell_max * 2 * np.pi
        itp1 = np.cos(tmp)
        itp2 = np.sin(tmp)

        mass = atom.get_masses()
        itp1_av = mass @ itp1 / mass.sum()
        itp2_av = mass @ itp2 / mass.sum()
        theta_av = np.arctan2(-itp2_av, -itp1_av) + np.pi
        res = cell_max * theta_av / 2 / np.pi
        return res

    def _find_axis_center_of_nanotube(
        self, atom: ase.atoms.Atoms
    ) -> ase.atoms.Atoms:
        """remove the center of structure to (x,y):(0,0)
        Args:
            atom: initial structure

        Returns: centralized structure

        """
        n_st = atom.copy()
        center = self._get_center_of_mass_periodic(atom)
        pos = (
            np.remainder(atom.get_scaled_positions() - center + 0.5, [1, 1, 1])
            @ atom.cell
        )

        atoms = Atoms(
            cell=n_st.cell,
            numbers=n_st.numbers,
            positions=pos,
        )
        return atoms

    # def get_symmetry_operations(self):
    #     generators = [op.affine_matrix for op in self.symmops if not np.allclose(op.affine_matrix, np.eye(4))]
    #     ops = brute_force_generate_group(generators, self.tol)
    #     ops_sym = [SymmOp(op) for op in ops]
    #     return ops_sym

    def get_generators(self):
        generators = [
            op.affine_matrix
            for op in self.symmops
            if not np.allclose(op.affine_matrix, np.eye(4))
        ]
        return generators


def get_symcell(monomer: Atoms) -> Atoms:
    apg = LineGroupAnalyzer(monomer)
    equ = list(apg.get_equivalent_atoms()["eq_sets"].keys())
    return monomer[equ]


def main():
    parser = argparse.ArgumentParser(
        description="Detect the line group of a system"
    )
    parser.add_argument(
        "filename", help="path to the file from which coordinates will be read"
    )
    parser.add_argument(
        "--enable_pg",
        action="store_true",
        help="open the detection of point group",
    )
    parser.add_argument(
        "--enable_log",
        action="store_true",
        help="Enable the output of detection process",
    )

    args = parser.parse_args()
    point_group_ind = args.enable_pg
    flag_log = args.enable_log
    if flag_log:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    st_name = args.filename
    st = read(st_name)

    mol = Molecule(species=st.numbers, coords=st.positions)
    obj = LineGroupAnalyzer(mol)
    # apg = obj.get_pointgroup()
    apg = obj.sch_symbol
    print(" Axial point group: ", apg)

    if point_group_ind:
        obj2 = PointGroupAnalyzer(mol)
        pg2 = obj2.get_pointgroup()
        print(" Point group: ", pg2)


if __name__ == "__main__":
    main()
