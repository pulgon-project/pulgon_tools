import argparse
from pdb import set_trace

import numpy as np
import pretty_errors
from ase.data import atomic_masses
from ase.io import read
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer


class LineGroupAnalyzer(PointGroupAnalyzer):
    """A class to analyze the axial point group of a molecule (based on pymatgen:PointGroupAnalyzer)

    The general outline of the algorithm is as follows:

    1. Specify z axis as the rotation axis, detect the rotational symmetry.
    2. If the rotational symmetry exist, detect the R2 axes perpendicular to z axis.
       - If R2 axes exist, it's a dihedral group (Dnh, Dnd).
       - If R2 axes does not exist, it's a dihedral group (Cnh, Cnv, S2n).
    3. If the rotational symmetry does not exist, only possible point groups are C1, Cs and Ci.
    """

    def __init__(
        self, mol, tolerance=0.3, eigen_tolerance=0.01, matrix_tolerance=0.1
    ):
        """The default settings are usually sufficient. (Totally the same with PointGroupAnalyzer)

        Args:
            mol (Molecule): Molecule to determine point group for.
            tolerance (float): Distance tolerance to consider sites as
                symmetrically equivalent. Defaults to 0.3 Angstrom.
            eigen_tolerance (float): Tolerance to compare eigen values of
                the inertia tensor. Defaults to 0.01.
            matrix_tolerance (float): Tolerance used to generate the full set of
                symmetry operations of the point group.
        """
        self.mol = mol
        self.centered_mol = mol.get_centered_molecule()
        self.tol = tolerance
        self.eig_tol = eigen_tolerance
        self.mat_tol = matrix_tolerance
        self._analyze()
        if self.sch_symbol in ["C1v", "C1h"]:
            self.sch_symbol = "Cs"

    def _analyze(self):
        """Rewrite the _analyze method, calculate the axial point group elements."""
        inertia_tensor = _inertia_tensor(self)
        _, eigvecs = np.linalg.eigh(inertia_tensor)
        self.principal_axes = eigvecs.T  # only be used in _proc_no_rot_sym

        self.rot_sym = []
        self.symmops = [SymmOp(np.eye(4))]

        z_axis = np.array([0, 0, 1])
        self._check_rot_sym(z_axis)

        if len(self.rot_sym) > 0:
            self._check_perpendicular_r2_axis(z_axis)

        if len(self.rot_sym) >= 2:
            self._proc_dihedral()
        elif len(self.rot_sym) == 1:
            self._proc_cyclic()
        else:
            self._proc_no_rot_sym()


def _inertia_tensor(self):
    """

    Returns: inertia_tensor of the molecular

    """
    weights = np.array([site.species.weight for site in self.centered_mol])
    Ixx = np.sum(
        weights * np.sum(self.centered_mol.cart_coords[:, 1:] ** 2, axis=1)
    )
    Iyy = np.sum(
        weights * np.sum(self.centered_mol.cart_coords[:, [0, 2]] ** 2, axis=1)
    )
    Izz = np.sum(
        weights * np.sum(self.centered_mol.cart_coords[:, :2] ** 2, axis=1)
    )
    Ixy = -1 * np.sum(
        weights
        * self.centered_mol.cart_coords[:, 0]
        * self.centered_mol.cart_coords[:, 1]
    )
    Ixz = -1 * np.sum(
        weights
        * self.centered_mol.cart_coords[:, 0]
        * self.centered_mol.cart_coords[:, 2]
    )
    Iyz = -1 * np.sum(
        weights
        * self.centered_mol.cart_coords[:, 1]
        * self.centered_mol.cart_coords[:, 2]
    )
    inertia_tensor = np.array(
        [[Ixx, Ixy, Izz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]
    )

    total_inertia = np.sum(
        weights * np.sum(self.centered_mol.cart_coords**2, axis=1)
    )
    inertia_tensor = inertia_tensor / total_inertia
    return inertia_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Try to detect the line group of a system"
    )
    parser.add_argument(
        "filename", help="path to the file from which coordinates will be read"
    )
    parser.add_argument(
        "--enable_pg",
        action="store_true",
        help="open the detection of point group",
    )
    args = parser.parse_args()

    point_group_ind = args.enable_pg
    # set_trace()

    st_name = args.filename
    st = read(st_name)

    mol = Molecule(species=st.numbers, coords=st.positions)
    obj1 = LineGroupAnalyzer(mol)
    pg1 = obj1.get_pointgroup()
    print(" Axial point group: ", pg1)

    if point_group_ind:
        obj2 = PointGroupAnalyzer(mol)
        pg2 = obj2.get_pointgroup()
        print(" Point group: ", pg2)


if __name__ == "__main__":
    main()
