import numpy as np
import sympy
from ase import Atoms
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value
from pymatgen.core.operations import SymmOp
from scipy.optimize import fsolve

from pulgon_tools.utils import Cn, brute_force_generate_group


def cyl2car(cyl):
    car = np.array([cyl[1] * np.cos(cyl[0]), cyl[1] * np.sin(cyl[0]), cyl[2]])
    return car


def helical_group_analysis(a1, a2, n1, n2):
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


def bond_constrains_equations(
    variables, pos_cyl1, pos_cyl2, pos_cyl3, pos_cyl4, bond_length
):
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

    initial_guess = [0, 0, 0]
    solutions1 = fsolve(
        bond_constrains_equations,
        initial_guess,
        args=(pos_cyl0, pos_cyl2, pos_cyl3, pos_cyl4, bond_length),
    )
    solutions2 = fsolve(
        bond_constrains_equations,
        initial_guess,
        args=(pos_cyl1, pos_cyl2, pos_cyl3, pos_cyl4, bond_length),
    )

    pos_cyl = np.array(
        [pos_cyl2, pos_cyl0 + solutions1, pos_cyl1 + solutions2]
    )
    numbers = np.array([symbol1, symbol2, symbol2])
    return pos_cyl, numbers


def get_nanotube_from_n1n2(n1, n2, symbol1, symbol2, L1, bond_length, delta_Z):
    a1 = L1 * np.array([1, 0])
    a2 = np.array([L1 * np.cos(np.pi / 3), L1 * np.sin(np.pi / 3)])
    q, f, r, R, Ch, t, t1, t2, n_gcd = helical_group_analysis(a1, a2, n1, n2)

    coo_cyl, numbers = generate_symcell_and_linegroup_elements(
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
                if judge == False:
                    pos_car = np.vstack((pos_car, tmp))
    numbers = np.tile(numbers, len(ops_sym))
    pos_car = np.array(pos_car)

    scaled = np.round(
        np.remainder(np.dot(pos_car, np.linalg.inv(cell)), [1, 1, 1]), 10
    )
    scaled, index = np.unique(scaled, axis=0, return_index=True)
    numbers = numbers[index]
    new_atom = Atoms(scaled_positions=scaled, cell=cell, numbers=numbers)
    return new_atom


if __name__ == "__main__":
    ##################### single layer ###################
    # n1, n2 = 24, 0
    # symbol1, symbol2 = 42, 16     # Mo, S
    # name = "MoS2"
    # # symbol1, symbol2 = 74, 16     # W, S
    # # name = "WS2"
    # atom1 = read_vasp("poscar-preoptimization/poscar_monolayer_%s.vasp" % name)
    # L1 = np.linalg.norm(atom1.cell[0])       # lattice parameter factor
    # delta_Z = abs((atom1.positions - atom1.positions[2])[0, 2])      # the distance between different layer in 2D
    # bond_length = np.linalg.norm((atom1.positions[0] - atom1.positions[2]))
    # new_atom = get_nanotube_from_n1n2(n1, n2, symbol1, symbol2, L1, bond_length, delta_Z)
    # pos = (new_atom.get_scaled_positions()-[0.5,0.5,0]) % 1 @ new_atom.cell
    # new_atom = Atoms(positions=pos, cell=new_atom.cell, numbers=new_atom.numbers)
    # write_vasp("poscar-preoptimization/poscar_%s_%d_%d.vasp" % (name, n1, n2), new_atom, direct=True, sort=True)
    ####################### multi layer ######################
    # n1,n2 = 13, 0
    # symbol1, symbol2 = 74, 16     # W, S
    # name1 = "WS2"
    # atom1 = read_vasp("poscar-preoptimization/poscar_monolayer_%s.vasp" % name1)
    # L1 = np.linalg.norm(atom1.cell[0])     # lattice pacmeter factor
    # delta_Z1 = abs((atom1.positions - atom1.positions[2])[0, 2])
    # bond_length1 = np.linalg.norm((atom1.positions[0] - atom1.positions[2]))
    #
    # new_atom1 = get_nanotube_from_n1n2(n1,n2,symbol1,symbol2, L1, bond_length1, delta_Z1)
    # n3,n4 = 26, 0
    # symbol1, symbol2 = 42, 16     # Mo, S
    # name2 = "MoS2"
    # atom2 = read_vasp("poscar-preoptimization/poscar_monolayer_%s.vasp" % name2)
    # L2 = np.linalg.norm(atom2.cell[0])     # lattice pacmeter factor
    # delta_Z2 = abs((atom2.positions - atom2.positions[2])[0, 2])
    # bond_length2 = np.linalg.norm((atom2.positions[0] - atom2.positions[2]))
    #
    # new_atom2 = get_nanotube_from_n1n2(n3,n4,symbol1,symbol2,L2, bond_length2, delta_Z2)
    # name = "WS2-%dx%d-MoS2-%dx%d" % (n1,n2,n3,n4)
    #
    # pos1 = (new_atom1.get_scaled_positions()-[0.5,0.5,0]) % 1 @ new_atom1.cell
    # pos2 = (new_atom2.get_scaled_positions()-[0.5,0.5,0]) % 1 @ new_atom2.cell
    #
    # tmp = np.concatenate(((new_atom2.cell - new_atom1.cell)[[0,1],[0,1]]/2, [0]))
    # pos1 = pos1 + tmp
    #
    # pos = np.vstack((pos1, pos2))
    # cell = new_atom2.cell
    # numbers = np.concatenate((new_atom1.numbers, new_atom2.numbers), axis=0)
    # new_atom = Atoms(positions=pos, cell=cell, numbers=numbers)
    # write_vasp("poscar-preoptimization/poscar_%s.vasp" % (name), new_atom, direct=True, sort=True)
    #######################################################
    pass
