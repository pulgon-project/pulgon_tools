import numpy as np
import pytest
from ase.io.vasp import write_vasp
from ipdb import set_trace

from pulgon_tools_wip.generate_structures import (
    Cn,
    S2n,
    U,
    U_d,
    dimino,
    generate_line_group_structure,
    sigmaH,
    sigmaV,
)


def pre_processing(pos_cylin, generators):
    if pos_cylin.ndim == 1:
        pos = np.array(
            [
                pos_cylin[0] * np.cos(pos_cylin[1]),
                pos_cylin[0] * np.sin(pos_cylin[1]),
                pos_cylin[2],
            ]
        )
    else:
        pos = np.array(
            [
                pos_cylin[:, 0] * np.cos(pos_cylin[:, 1]),
                pos_cylin[:, 0] * np.sin(pos_cylin[:, 1]),
                pos_cylin[:, 2],
            ]
        )
        pos = pos.T
    rot_sym = dimino(generators, symec=4)
    monomer_pos = []
    for sym in rot_sym:
        if pos.ndim == 1:
            monomer_pos.append(np.dot(sym, pos.reshape(pos.shape[0], 1)).T[0])
        else:
            monomer_pos.extend([np.dot(sym, line) for line in pos])
    monomer_pos = np.array(monomer_pos)
    return monomer_pos


def test_st3():
    """(I|q),Cn,sigmaH"""
    """(I|a),Cn,sigmaV"""
    motif = np.array([3, np.pi / 24, 1])
    generators = np.array([Cn(6), sigmaV()])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)

    write_vasp("st6.vasp", st)
    set_trace()


if __name__ == "__main__":
    test_st3()
