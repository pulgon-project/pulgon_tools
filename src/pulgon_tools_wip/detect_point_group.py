import argparse
from pdb import set_trace

import numpy as np
import pretty_errors
from ase.io import read
from pymatgen.core import Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from pulgon_tools_wip.generate_structures import sigmaV


class LineGroupAnalyzer(PointGroupAnalyzer):
    def __init__(
        self, mol, tolerance=0.3, eigen_tolerance=0.01, matrix_tolerance=0.1
    ):
        self.mol = mol
        self.centered_mol = mol.get_centered_molecule()
        self.tol = tolerance
        self.eig_tol = eigen_tolerance
        self.mat_tol = matrix_tolerance
        self._analyze()
        if self.sch_symbol in ["C1v", "C1h"]:
            self.sch_symbol = "Cs"

    def _analyze(self):
        inertia_tensor = np.zeros((3, 3))
        total_inertia = 0
        for site in self.centered_mol:
            c = site.coords
            wt = site.species.weight
            for i in range(3):
                inertia_tensor[i, i] += wt * (
                    c[(i + 1) % 3] ** 2 + c[(i + 2) % 3] ** 2
                )
            for i, j in [(0, 1), (1, 2), (0, 2)]:
                inertia_tensor[i, j] += -wt * c[i] * c[j]
                inertia_tensor[j, i] += -wt * c[j] * c[i]
            total_inertia += wt * np.dot(c, c)
        inertia_tensor /= total_inertia
        eigvals, eigvecs = np.linalg.eig(inertia_tensor)
        self.principal_axes = eigvecs.T

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
