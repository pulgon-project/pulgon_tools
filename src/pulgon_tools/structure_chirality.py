# Copyright 2023-2026 The PULGON Project Developers
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

import numpy as np
import sympy
from ase import Atoms
from ase.io.vasp import write_vasp
from pulp import LpMinimize, LpProblem, LpVariable, value
from pymatgen.core.operations import SymmOp
from scipy.optimize import fsolve

from pulgon_tools.utils import Cn, brute_force_generate_group


def cyl2car(cyl):
    """Convert cylindrical coordinates (phi, r, z) to Cartesian (x, y, z).

    Args:
        cyl: array-like of [phi, r, z] in cylindrical coordinates.

    Returns:
        np.ndarray of [x, y, z] in Cartesian coordinates.
    """
    car = np.array([cyl[1] * np.cos(cyl[0]), cyl[1] * np.sin(cyl[0]), cyl[2]])
    return car


def helical_group_analysis(a1, a2, n1, n2, L1):
    """Compute helical group parameters for a chiral nanotube (n1, n2).

    Derives the screw-axis order q, translational pitch f, tube radius r,
    helical quantum number R, chiral vector direction Ch, translational
    period t, translation indices (t1, t2), and GCD of (n1, n2).

    Args:
        a1: first 2D lattice vector of the hexagonal sheet.
        a2: second 2D lattice vector of the hexagonal sheet.
        n1: first chiral index.
        n2: second chiral index.
        L1: lattice constant (length of a1).

    Returns:
        symmetry indices: (q, f, r, R, Ch, t, t1, t2, n_gcd).
    """
    n_gcd = np.gcd(n1, n2)
    n1_tilde = int(n1 / n_gcd)
    n2_tilde = int(n2 / n_gcd)

    t_gcd = np.gcd(2 * n2_tilde + n1_tilde, 2 * n1_tilde + n2_tilde)
    t1 = -int((2 * n2_tilde + n1_tilde) / t_gcd)
    t2 = int((2 * n1_tilde + n2_tilde) / t_gcd)
    t = np.linalg.norm(t1 * a1 + t2 * a2)
    q_tilde = int(np.linalg.det([[n1_tilde, n2_tilde], [t1, t2]]))
    q = n_gcd * q_tilde

    f = t / q_tilde
    D = (
        L1
        / np.pi
        * n_gcd
        * np.sqrt(n1_tilde**2 + n1_tilde * n2_tilde + n2_tilde**2)
    )
    r = D / 2

    prob = LpProblem("IntegerProgrammingExample", LpMinimize)
    h1 = LpVariable("h1", lowBound=1, cat="Integer")
    h2 = LpVariable("h2", lowBound=1, cat="Integer")
    prob += h1 + h2
    prob += n1_tilde * h2 - n2_tilde * h1 == 1
    prob.solve()

    h1 = int(value(h1))
    h2 = int(value(h2))
    R = h1 * t2 - h2 * t1

    p_tilde = R ** (sympy.totient(q_tilde) - 1)
    p = n_gcd * p_tilde

    Ch = n1 * a1 + n2 * a2
    Ch = Ch / np.linalg.norm(Ch)
    return q, f, r, R, Ch, t, t1, t2, n_gcd


def bond_constraints_equations(
    variables, pos_cyl1, pos_cyl2, pos_cyl3, pos_cyl4, bond_length
):
    """Nonlinear equations enforcing equal bond lengths on a nanotube surface.

    Solves for (del_phi, del_r, del_z) corrections to pos_cyl1 such that
    its distances to pos_cyl2, pos_cyl3, and pos_cyl4 are all equal to
    bond_length. Used as input to scipy.optimize.fsolve.

    Args:
        variables: array of [del_phi, del_r, del_z] corrections.
        pos_cyl1: cylindrical coordinates [phi, r, z] of the atom to adjust.
        pos_cyl2: cylindrical coordinates of the first neighbor.
        pos_cyl3: cylindrical coordinates of the second neighbor.
        pos_cyl4: cylindrical coordinates of the third neighbor.
        bond_length: target bond length.

    Returns:
        list of 3 residual equations [eq1, eq2, eq3].
    """
    del_phi, del_r, del_z = variables
    pos_car1 = np.array(
        [
            (pos_cyl1[1] + del_r) * np.cos(pos_cyl1[0] + del_phi),
            (pos_cyl1[1] + del_r) * np.sin(pos_cyl1[0] + del_phi),
            pos_cyl1[2] + del_z,
        ]
    )
    pos_car2 = np.array(
        [
            pos_cyl2[1] * np.cos(pos_cyl2[0]),
            pos_cyl2[1] * np.sin(pos_cyl2[0]),
            pos_cyl2[2],
        ]
    )
    pos_car3 = np.array(
        [
            pos_cyl3[1] * np.cos(pos_cyl3[0]),
            pos_cyl3[1] * np.sin(pos_cyl3[0]),
            pos_cyl3[2],
        ]
    )
    pos_car4 = np.array(
        [
            pos_cyl4[1] * np.cos(pos_cyl4[0]),
            pos_cyl4[1] * np.sin(pos_cyl4[0]),
            pos_cyl4[2],
        ]
    )
    eq1 = np.linalg.norm(pos_car1 - pos_car2) - np.linalg.norm(
        pos_car1 - pos_car3
    )
    eq2 = np.linalg.norm(pos_car1 - pos_car2) - np.linalg.norm(
        pos_car1 - pos_car4
    )
    eq3 = np.linalg.norm(pos_car1 - pos_car2) - bond_length
    return [eq1, eq2, eq3]


def generate_symcell_and_linegroup_elements(
    a1,
    a2,
    Ch,
    t1,
    t2,
    r,
    bond_length,
    delta_Z,
    symbol1=74,
    symbol2=16,
    tol_round=10,
):
    """Generate the symmetry cell (monomer) for a MoS2-type chiral nanotube.

    Maps flat-sheet atomic positions onto the tube surface in cylindrical
    coordinates, then optimizes chalcogen positions to satisfy bond-length
    constraints via fsolve.

    Args:
        a1: first 2D lattice vector.
        a2: second 2D lattice vector.
        Ch: unit chiral vector direction.
        t1: first translational index.
        t2: second translational index.
        r: tube radius.
        bond_length: target bond length between atoms.
        delta_Z: vertical offset of chalcogen atoms from the metal layer.
        symbol1: atomic number of the metal atom (default 74, W).
        symbol2: atomic number of the chalcogen atom (default 16, S).
        tol_round: decimal precision for rounding intermediate values.

    Returns:
        tuple of (pos_cyl, symbols) where pos_cyl is an (N, 3) array of
        cylindrical coordinates [phi, r, z] and symbols is an (N,) array
        of atomic numbers.
    """
    pos1 = 1 / 3 * a1 + 1 / 3 * a2
    pos2 = 2 / 3 * a1 + 2 / 3 * a2
    pos_auxiliary1 = 4 / 3 * a1 + 1 / 3 * a2
    pos_auxiliary2 = 1 / 3 * a1 + 4 / 3 * a2

    r1 = r + delta_Z
    r2 = r - delta_Z

    z1 = np.sqrt(
        np.round(np.linalg.norm(pos1) ** 2 - np.dot(pos1, Ch) ** 2, tol_round)
    )
    z2 = np.sqrt(
        np.round(np.linalg.norm(pos2) ** 2 - np.dot(pos2, Ch) ** 2, tol_round)
    )
    z_auxiliary1 = np.sign((t1 * a1 + t2 * a2) @ pos_auxiliary1) * np.sqrt(
        np.linalg.norm(pos_auxiliary1) ** 2 - np.dot(pos_auxiliary1, Ch) ** 2
    )
    z_auxiliary2 = np.sign((t1 * a1 + t2 * a2) @ pos_auxiliary2) * np.sqrt(
        np.linalg.norm(pos_auxiliary2) ** 2 - np.dot(pos_auxiliary2, Ch) ** 2
    )

    phi1 = np.sqrt(np.linalg.norm(pos1) ** 2 - z1**2) / r
    phi2 = np.sqrt(np.linalg.norm(pos2) ** 2 - z2**2) / r
    phi3 = np.sqrt(np.linalg.norm(pos2) ** 2 - z2**2) / r
    phi_auxiliary1 = (
        np.sqrt(np.linalg.norm(pos_auxiliary1) ** 2 - z_auxiliary1**2) / r
    )
    phi_auxiliary2 = (
        np.sqrt(np.linalg.norm(pos_auxiliary2) ** 2 - z_auxiliary2**2) / r
    )

    pos_cyl0 = [phi2, r1, z2]
    pos_cyl1 = [phi3, r2, z2]
    pos_cyl2 = [phi1, r, z1]

    pos_cyl3 = [phi_auxiliary1, r, z_auxiliary1]
    pos_cyl4 = [phi_auxiliary2, r, z_auxiliary2]

    initial_guess = np.array([0, 0, 0])
    solutions1 = fsolve(
        bond_constraints_equations,
        initial_guess,
        args=(pos_cyl0, pos_cyl2, pos_cyl3, pos_cyl4, bond_length),
    )
    solutions2 = fsolve(
        bond_constraints_equations,
        initial_guess,
        args=(pos_cyl1, pos_cyl2, pos_cyl3, pos_cyl4, bond_length),
    )

    pos_cyl = np.array(
        [pos_cyl2, pos_cyl0 + solutions1, pos_cyl1 + solutions2]
    )
    symbols = np.array([symbol1, symbol2, symbol2])
    return pos_cyl, symbols


def get_nanotube_from_n1n2(n1, n2, symbol1, symbol2, bond_length, delta_Z):
    """Build a complete MoS2-type chiral nanotube from chiral indices (n1, n2).

    Computes helical group parameters, generates the symmetry cell, applies
    all line group operations, and returns a deduplicated ASE Atoms object.

    Args:
        n1: first chiral index.
        n2: second chiral index.
        symbol1: atomic number or symbol of the metal atom.
        symbol2: atomic number or symbol of the chalcogen atom.
        bond_length: metal-chalcogen bond length in Angstrom.
        delta_Z: vertical offset of chalcogen atoms from the metal layer.

    Returns:
        ase.Atoms: the nanotube structure with periodic boundary along z.
    """
    L1 = np.sqrt(bond_length**2 - delta_Z**2) * np.sqrt(
        2 - 2 * np.cos(2 * np.pi / 3)
    )  # the length of hex lattice
    a1 = L1 * np.array([1, 0])
    a2 = np.array([L1 * np.cos(np.pi / 3), L1 * np.sin(np.pi / 3)])
    q, f, r, R, Ch, t, t1, t2, n_gcd = helical_group_analysis(
        a1, a2, n1, n2, L1
    )

    coo_cyl, symbols = generate_symcell_and_linegroup_elements(
        a1,
        a2,
        Ch,
        t1,
        t2,
        r,
        bond_length=bond_length,
        delta_Z=delta_Z,
        symbol1=symbol1,
        symbol2=symbol2,
    )
    car_x = coo_cyl[:, 1] * np.cos(coo_cyl[:, 0])
    car_y = coo_cyl[:, 1] * np.sin(coo_cyl[:, 0])
    pos = np.hstack(
        (car_x[np.newaxis].T, car_y[np.newaxis].T, coo_cyl[:, 2][np.newaxis].T)
    )

    distance = np.max(pos[:, :2])
    cell = np.array(
        [[distance * 4.5, 0, 0], [0, distance * 4.5, 0], [0, 0, t]]
    )

    generators = [
        SymmOp.from_rotation_and_translation(
            rotation_matrix=Cn(n_gcd), translation_vec=[0, 0, 0]
        ).affine_matrix,
        SymmOp.from_rotation_and_translation(
            rotation_matrix=Cn(q / R), translation_vec=[0, 0, 1 / int(t / f)]
        ).affine_matrix,
    ]
    ops = brute_force_generate_group(generators, symec=1e-2)

    pos_car = np.array([])
    ops_sym = [
        SymmOp.from_rotation_and_translation(
            rotation_matrix=line[:3, :3], translation_vec=line[:3, 3] * t
        )
        for line in ops
    ]
    for op in ops_sym:
        for line in pos:
            tmp = op.operate(line)
            if pos_car.size == 0:
                pos_car = tmp[np.newaxis]
            else:
                judge = (
                    np.sqrt((pos_car - tmp) ** 2).sum(axis=1) < 1e-8
                ).any()
                if not judge:
                    pos_car = np.vstack((pos_car, tmp))
    symbols = np.tile(symbols, len(ops_sym))
    pos_car = np.array(pos_car)

    scaled = np.round(
        np.remainder(np.dot(pos_car, np.linalg.inv(cell)), [1, 1, 1]), 10
    )
    scaled, index = np.unique(scaled, axis=0, return_index=True)
    symbols = symbols[index]
    new_atom = Atoms(scaled_positions=scaled, cell=cell, symbols=symbols)
    return new_atom


def main():
    """CLI entry point for generating a chiral nanotube and saving as POSCAR."""
    parser = argparse.ArgumentParser(
        description="generating chiral nanotube structure from (n1, n2) indices"
    )

    parser.add_argument(
        "-c",
        "--chirality",
        default=(10, 10),
    )

    parser.add_argument(
        "-b",
        "--symbol",
        default=("Mo", "S"),
    )

    parser.add_argument("-l", "--bond_length", default=2.43, type=float)

    parser.add_argument("-d", "--delta_Z", default=1.57, type=float)

    parser.add_argument(
        "-s",
        "--st_name",
        type=str,
        default="POSCAR",
        help="the saved file name",
    )

    args = parser.parse_args()

    n1, n2 = eval(args.chirality)[0], eval(args.chirality)[1]

    symbol1, symbol2 = eval(args.symbol)[0], eval(args.symbol)[1]
    bond_length = args.bond_length
    delta_Z = args.delta_Z
    filename = args.st_name

    new_atom = get_nanotube_from_n1n2(
        n1, n2, symbol1, symbol2, bond_length, delta_Z
    )
    pos = (new_atom.get_scaled_positions() - [0.5, 0.5, 0]) % 1 @ new_atom.cell
    new_atom = Atoms(
        positions=pos, cell=new_atom.cell, numbers=new_atom.numbers
    )
    write_vasp(filename, new_atom, direct=True, sort=True)


if __name__ == "__main__":
    main()
