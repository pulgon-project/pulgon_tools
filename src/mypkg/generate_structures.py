import argparse
import copy

import numpy as np
from ase import Atoms
from ase.io.vasp import write_vasp

parser = argparse.ArgumentParser(description="generating line group structure")

parser.add_argument(
    "-m",
    "--motif",
    default=[[3, np.pi / 24, 0.6], [2.2, np.pi / 24, 0.8]],
    help="the Cylindrical coordinates of initial atom position",
)

parser.add_argument(
    "-g",
    "--generators",
    default=["Cn(6)", "sigmaV()"],
    help="the point group generator of monomer",
)

parser.add_argument(
    "-c",
    "--cyclic",
    default={"T_Q": [6, 1.5]},
    help="The generalized translation group. For T_Q the first parameter is Q and the second parameter is f."
    " For T_V the parameter is f",
)

parser.add_argument(
    "-s",
    "--st_name",
    type=str,
    default="poscar.vasp",
    help="the saved file name",
)

args = parser.parse_args()


def e():
    """
    Returns: identity matrix
    """
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return mat


def Cn(n):
    """
    Args:
        n: rotate 2*pi/n

    Returns: rotation matrix
    """
    mat = np.array(
        [
            [np.cos(2 * np.pi / n), -np.sin(2 * np.pi / n), 0],
            [np.sin(2 * np.pi / n), np.cos(2 * np.pi / n), 0],
            [0, 0, 1],
        ]
    )
    return mat


def sigmaV():
    """

    Returns: mirror symmetric matrix

    """
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return mat


def sigmaH():
    """

    Returns: mirror symmetric matrix about x-y plane

    """
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    return mat


def U():
    """

    Returns: A symmetric matrix about the x-axis

    """
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return mat


def U_d(fid):
    """

    Args:
        fid: the angle between symmetry axis d and axis x, d located in th x-y plane

    Returns: A symmetric matrix about the d-axis

    """
    mat = np.array(
        [
            [np.cos(2 * fid), np.sin(2 * fid), 0],
            [np.sin(2 * fid), -np.cos(2 * fid), 0],
            [0, 0, -1],
        ]
    )
    return mat


def S2n(n):
    """
    Args:
        n: dihedral group, rotate 2*pi/n

    Returns: rotation and mirror matrix

    """
    mat = np.array(
        [
            [np.cos(np.pi / n), -np.sin(np.pi / n), 0],
            [np.sin(np.pi / n), np.cos(np.pi / n), 0],
            [0, 0, -1],
        ]
    )
    return mat


def T_Q(Q, f, pos):
    """

    Args:
        Q: rotate 2*pi/n
        f: movement distance along z axis
        pos: monomer positions

    Returns: the positions of all atoms

    """
    if pos.ndim == 2:
        pos = np.dot(Cn(Q), pos.T)
    else:
        print("error with dim")
    pos = pos.T
    pos[:, 2] = pos[:, 2] + f
    return pos


def T_v(f, pos):
    """

    Args:
        f: movement distance along z axis
        pos: monomer positions

    Returns: the positions of all atoms

    """
    if pos.ndim == 2:
        pos = np.dot(sigmaV(), pos.T)
    else:
        print("error with dim")
    pos = pos.T
    pos[:, 2] = pos[:, 2] + f
    return pos


def dimino(generators, symec=4):
    """

    Args:
        generators: the generators of point group
        symec: system precision

    Returns: all the group elements

    """
    G = generators
    g, g1 = copy.deepcopy(G[0]), copy.deepcopy(G[0])
    L = np.array([e()])
    while not (np.round(g, symec) == e()).all():
        L = np.vstack((L, [g]))
        g = np.dot(g, g1)
    L = np.round(L, symec)

    for ii in range(len(G)):
        C = np.array([e()])
        L1 = copy.deepcopy(L)
        more = True
        while more:
            more = False
            for g in list(C):
                for s in G[: ii + 1]:
                    sg = np.round(np.dot(s, g), 4)
                    itp = (sg == L).all(axis=1).all(axis=1).any()
                    if not itp:
                        if C.ndim == 3:
                            C = np.vstack((C, [sg]))
                        else:
                            C = np.array((C, sg))
                        if L.ndim == 3:
                            L = np.vstack(
                                (L, np.array([np.dot(sg, t) for t in L1]))
                            )
                        else:
                            L = np.array(
                                L, np.array([np.dot(sg, t) for t in L1])
                            )
                        more = True
    return L


def change_center(st1):
    """

    Args:
        st1: an ase.atom structure

    Returns: an ase.atom structure with z axis located in the cell center

    """
    st1_pos = st1.get_scaled_positions()
    st2_pos = st1_pos[:, :2] + 0.5
    tmp = np.modf(st2_pos)[0]
    tmp1 = st1_pos[:, 2]
    tmp1 = tmp1.reshape(tmp1.shape[0], 1)

    st2 = copy.deepcopy(st1)
    st2.positions = np.dot(np.hstack((tmp, tmp1)), st2.cell)
    return st2


def generate_line_group_structure(monomer_pos, cyclic_group):
    """

    Args:
        monomer_pos: the positions of monomer
        cyclic_group: the generalized translation group

    Returns: the final structure after all symmetry operations

    """
    all_pos = copy.deepcopy(monomer_pos)
    if list(cyclic_group.keys())[0] == "T_Q":
        Q = cyclic_group["T_Q"][0]
        f = cyclic_group["T_Q"][1]
        for ii in range(np.ceil(Q).astype(np.int32)):
            all_pos = np.vstack((all_pos, T_Q(Q, f, all_pos)))
        all_pos = np.unique(np.round(all_pos, 4), axis=0)
        A = Q * f

    elif list(cyclic_group.keys())[0] == "T_V":
        f = cyclic_group["T_V"]
        for ii in range(2):
            all_pos = np.vstack((all_pos, T_v(f, all_pos)))
        all_pos = np.unique(np.round(all_pos, 4), axis=0)
        A = 2 * f
    else:
        print("A error input about cyclic_group")

    p0 = np.max(np.sqrt(all_pos[:, 0] ** 2 + all_pos[:, 1] ** 2))
    cell = np.array([[p0 * 3, 0, 0], [0, p0 * 3, 0], [0, 0, A]])

    st1 = Atoms(symbols="C" + str(len(all_pos)), positions=all_pos, cell=cell)
    st2 = change_center(st1)  # change the axis center to cell center
    return st2


def main():
    pos_cylin = np.array(eval(args.motif))
    if pos_cylin.ndim == 1:
        pos = pos_cylin.reshape(pos_cylin.shape[0], 1)
    else:
        pos = np.array(
            [
                pos_cylin[:, 0] * np.cos(pos_cylin[:, 1]),
                pos_cylin[:, 0] * np.sin(pos_cylin[:, 1]),
                pos_cylin[:, 2],
            ]
        )
        pos = pos.T
    generators = np.array([eval(tmp) for tmp in eval(args.generators)])
    cg = eval(args.cyclic)
    # set_trace()
    st_name = args.st_name

    rot_sym = dimino(generators, symec=4)
    monomer_pos = []
    for sym in rot_sym:
        if pos.ndim == 1:
            monomer_pos.append(np.dot(sym, pos).T[0])
        else:
            monomer_pos.extend([np.dot(sym, line) for line in pos])
    monomer_pos = np.array(monomer_pos)

    st = generate_line_group_structure(monomer_pos, cg)
    write_vasp("%s" % st_name, st, direct=True, sort=True)


if __name__ == "__main__":
    main()
