import copy
import math
from pdb import set_trace

import numpy as np
import pretty_errors
from ase import Atoms
from ase.build import nanotube
from ase.io.vasp import read_vasp, write_vasp


def e():
    """
    Returns:

    """
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return mat


def Cn(n):
    mat = np.array(
        [
            [np.cos(2 * np.pi / n), -np.sin(2 * np.pi / n), 0],
            [np.sin(2 * np.pi / n), np.cos(2 * np.pi / n), 0],
            [0, 0, 1],
        ]
    )
    # mat = [[np.cos(2*np.pi/n), -np.sin(2*np.pi/n), 0], [np.sin(2*np.pi/n),np.cos(2*np.pi/n), 0], [0,0,1]]
    return mat


def sigmaV():
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return mat


def sigmaH():
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    return mat


def U():
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return mat


def U_d(fid):
    """

    Args:
        fid: the angle between symmetry axis d and axis x, d located in th x-y plane

    Returns:

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
        n:

    Returns:

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
    if pos.ndim == 2:
        pos = np.dot(Cn(Q), pos.T)
    else:
        print("error with dim")
    pos = pos.T
    pos[:, 2] = pos[:, 2] + f
    return pos


def T_v(a, pos):
    if pos.ndim == 2:
        pos = np.dot(sigmaV(), pos.T)
    else:
        print("error with dim")
    pos = pos.T
    pos[:, 2] = pos[:, 2] + a / 2
    return pos


def dimino(generators, symec=4):
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
    st1_pos = st1.get_scaled_positions()
    st2_pos = st1_pos[:, :2] + 0.5
    tmp = np.modf(st2_pos)[0]
    tmp1 = st1_pos[:, 2]
    tmp1 = tmp1.reshape(tmp1.shape[0], 1)

    st2 = copy.deepcopy(st1)
    st2.positions = np.dot(np.hstack((tmp, tmp1)), st2.cell)
    return st2


def test_example():
    """

    Returns:

    """
    p0, fi0 = 3, np.pi / 16

    x0, y0, z0 = p0 * np.cos(fi0), p0 * np.sin(fi0), 0.6
    pos = np.array([[x0], [y0], [z0]]).astype(np.float32)
    a, n = 6, 6

    # generate monomer (point group part)
    G = [Cn(n), U(), sigmaV()]
    rot_sym = dimino(G)

    monomer_pos = []
    for sym in rot_sym:
        monomer_pos.append(np.dot(sym, pos).T[0])

    monomer_pos = np.array(monomer_pos)
    all_pos = copy.deepcopy(monomer_pos)
    # set_trace()

    # Translation part
    for ii in range(2):
        all_pos = np.vstack((all_pos, T_Q(2 * n, a / 2, all_pos)))
    all_pos = np.unique(np.round(all_pos, 4), axis=0)

    h = a

    cell = np.array([[p0 * 3, 0, 0], [0, p0 * 3, 0], [0, 0, h]])
    # set_trace()
    x = all_pos[:, 0]
    y = all_pos[:, 1]
    z = all_pos[:, 2]
    pos = np.hstack(
        (
            x.reshape(x.shape[0], 1),
            y.reshape(y.shape[0], 1),
            z.reshape(z.shape[0], 1),
        )
    )

    st1 = Atoms(symbols="C" + str(len(pos)), positions=pos, cell=cell)
    st2 = change_center(st1)  # change the axis center to (0, 0, z)

    write_vasp("test.vasp", st2, direct=True, sort=True)


def main():
    test_example()


if __name__ == "__main__":
    main()
