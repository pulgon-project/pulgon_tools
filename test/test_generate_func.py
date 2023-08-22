import numpy as np
import pytest

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


@pytest.fixture(name="generators")
def fixture_point_group_generators():
    return [Cn(6), sigmaV(), U()]


def test_dimino(generators):
    assert len(dimino(generators)) == 24


def test_st1():
    """(Cq|f),Cn"""
    motif = np.array([2, 0, 0])
    generators = np.array([Cn(4)])
    cyclic = {"T_Q": [6, 1.5]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 12


def test_st2():
    """(I|q),S2n"""
    motif = np.array([3, 0, 1])
    generators = np.array([S2n(6)])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 18


def test_st3():
    """(I|q),Cn,sigmaH"""
    motif = np.array([2.5, 0, 1])
    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 18


def test_st4():
    """(C2n|f/2),Cn,sigmaH"""
    motif = np.array([3, 0, 0.6])
    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_Q": [12, 4]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 24


def test_st5():
    """(Cq|f),Cn,U"""
    motif = np.array([3, np.pi / 9, 0.5])
    generators = np.array([Cn(6), U()])
    cyclic = {"T_Q": [4, 4]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 24


def test_st6():
    """(I|a),Cn,sigmaV"""
    motif = np.array([3, np.pi / 24, 1])
    generators = np.array([Cn(6), sigmaV()])
    cyclic = {"T_Q": [1, 3]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 12


def test_st7():
    """(sigmaV|a/2),Cn"""
    motif = np.array([3, np.pi / 24, 1])
    generators = np.array([Cn(6)])
    cyclic = {"T_V": 1.5}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 12


def test_st8():
    """(C2n|a/2),Cn,sigmaV"""
    motif = np.array([3, np.pi / 24, 0])
    generators = np.array([Cn(6), sigmaV()])
    cyclic = {"T_Q": [12, 1.5]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 26


def test_st9():
    """(I|a),Cn,Ud,sigmaV"""
    motif = np.array([3, np.pi / 24, 0.6])
    generators = np.array([Cn(6), U_d(np.pi / 12), sigmaV()])
    cyclic = {"T_Q": [1, 4]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 24


def test_st10():
    """(sigmaV|a/2),S2n"""
    motif = np.array([3, np.pi / 18, 0.4])
    generators = np.array([S2n(6)])
    cyclic = {"T_V": 4}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 24


def test_st11():
    """(I|a),Cn,sigmaV"""
    motif = np.array([3, np.pi / 18, 0.6])
    generators = np.array([Cn(6), U(), sigmaV()])
    cyclic = {"T_Q": [1, 4]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 24


def test_st12():
    """(sigmaV|a),Cn,U,sigmaV"""
    motif = np.array([3, np.pi / 24, 0.5])
    generators = np.array([Cn(6), sigmaH()])
    cyclic = {"T_V": 2.5}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 24


def test_st13():
    """(C2n|a/2),Cn,U,sigmaV"""
    motif = np.array([3, np.pi / 16, 0.6])
    generators = np.array([Cn(6), U(), sigmaV()])
    cyclic = {"T_Q": [12, 3]}

    monomer_pos = pre_processing(motif, generators)
    st = generate_line_group_structure(monomer_pos, cyclic)
    assert len(st) == 48
