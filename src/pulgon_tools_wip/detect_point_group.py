import argparse
from pdb import set_trace

import numpy as np
import pretty_errors
from ase.io import read
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer


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
        mol: Molecule,
        tolerance: float = 0.3,
        eigen_tolerance: float = 0.01,
        matrix_tolerance: float = 0.1,
    ):
        """The default settings are usually sufficient. (Totally the same with PointGroupAnalyzer)

        Args:
            mol (Molecule): Molecule to determine point group.
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
        inertia_tensor = self._inertia_tensor()
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
