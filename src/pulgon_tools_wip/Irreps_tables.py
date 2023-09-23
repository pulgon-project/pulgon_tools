import typing
from pdb import set_trace

import matplotlib.pyplot as plt
import numpy as np
import pretty_errors
from utils import frac_range


def _cal_irrep_trace(irreps, symprec):
    character_tabel = []
    for irrep in irreps:
        tmp_character = []
        for tmp in irrep:

            if tmp.size == 1:
                if abs(tmp.imag) < symprec:
                    tmp = tmp.real
                if abs(tmp.real) < symprec:
                    tmp = complex(0, tmp.imag)
                    if tmp.imag == 0:
                        tmp = 0
                tmp_character.append(tmp)
            else:
                tmp = np.trace(tmp)
                if abs(tmp.imag) < symprec:
                    tmp = tmp.real
                if abs(tmp.real) < symprec:
                    tmp = complex(0, tmp.imag)
                    if abs(tmp.imag) < symprec:
                        tmp = 0
                tmp_character.append(tmp)
        character_tabel.append(tmp_character)
    return character_tabel


class CharacterDataset(typing.NamedTuple):
    """

    Args:
        index:
        quantum_number:
        character_table:
    """

    index: list[int]
    quantum_number: list[tuple]
    character_table: list


def line_group_1(
    q: int,
    r: int,
    a: float,
    f: float,
    n: int,
    k1: float,
    k2: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """TQ(f)Cn"""
    # label for line group family
    # row_labels = [r"$(C_{Q}|f)$", r"$C_{n}$"]
    # column_labels = [r"$_{k}A_{m}$", r"$_{\widetilde{k}}A_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if k1 <= -np.pi / a or k1 > np.pi / a:
        judge = False
        message.append("k1 not belong to (-pi/a,pi/a]")

    if k2 <= -np.pi / f or k2 > np.pi / f:
        judge = False
        message.append("k2 not belong to (-pi/f,pi/f]")

    m1 = frac_range(-q / 2, q / 2, left=False)
    m2 = frac_range(-n / 2, n / 2, left=False)

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                irrep1 = np.round(
                    [
                        np.exp(1j * (k1 * f + tmp_m1 * 2 * np.pi * r / q)),
                        np.exp(1j * tmp_m1 * 2 * np.pi / n),
                    ],
                    round_symprec,
                )
                irrep2 = np.round(
                    [np.exp(1j * k2 * f), np.exp(1j * tmp_m2 * 2 * np.pi / n)],
                    round_symprec,
                )
                irreps = [irrep1, irrep2]
                character_table.append(_cal_irrep_trace(irreps, symprec))
                quantum_number.append(((k1, tmp_m1), (k2, tmp_m2)))
                index.append(ind)
                ind = ind + 1
        # set_trace()
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_2(
    a: float,
    n: float,
    k1: float,
    k2: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """T(a)S2n"""
    # row_labels = [r"$(I|a)$", r"$\sigma_{h}C_{2n}$"]
    # column_labels = [r"$_{k}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1) >= symprec and abs(k1 - np.pi / a) > symprec:
        judge = False
        message.append("k1 not equal to 0 or pi/a")
    if k2 <= 0 or k2 >= np.pi:
        judge = False
        message.append("k2 not belong to (0,pi)")

    m1 = frac_range(-n / 2, n / 2, left=False)
    m2 = frac_range(-n / 2, n / 2, left=False)
    piH = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_piH in piH:
                    irrep1 = np.round(
                        [
                            np.exp(1j * k1 * a),
                            tmp_piH * np.exp(1j * tmp_m1 * np.pi / n),
                        ],
                        round_symprec,
                    )
                    irrep2 = np.round(
                        [
                            [
                                [np.exp(1j * k2 * a), 0],
                                [0, np.exp(-1j * k2 * a)],
                            ],
                            [[0, np.exp(1j * tmp_m2 * np.pi / n)], [1, 0]],
                        ],
                        round_symprec,
                    )
                    irreps = [irrep1, irrep2]
                    character_table.append(_cal_irrep_trace(irreps, symprec))
                    quantum_number.append(
                        ((k1, tmp_m1, tmp_piH), (k2, tmp_m2))
                    )
                    index.append(ind)
                    ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_3(
    a: float,
    n: float,
    k1: float,
    k2: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """T(a)Cnh"""
    # row_labels = [r"$(I|a)$", r"$C_{n}$", r"$\sigma_{h}$"]
    # column_labels = [r"$_{k}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1) >= symprec and abs(k1 - np.pi / a) > symprec:
        judge = False
        message.append("k1 not equal to 0 or pi/a")
    if k2 <= 0 or k2 >= np.pi:
        judge = False
        message.append("k2 not belong to (0,pi)")

    m1 = frac_range(-n / 2, n / 2, left=False)
    m2 = frac_range(-n / 2, n / 2, left=False)
    piH = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_piH in piH:
                    irrep1 = np.round(
                        [
                            np.exp(1j * k1 * a),
                            np.exp(1j * tmp_m1 * 2 * np.pi / n),
                            tmp_piH,
                        ],
                        round_symprec,
                    )
                    irrep2 = np.round(
                        [
                            [
                                [np.exp(1j * k2 * a), 0],
                                [0, np.exp(-1j * k2 * a)],
                            ],
                            [
                                [np.exp(1j * tmp_m2 * 2 * np.pi / n), 0],
                                [0, np.exp(1j * tmp_m2 * 2 * np.pi / n)],
                            ],
                            [[0, 1], [1, 0]],
                        ],
                        round_symprec,
                    )
                    irreps = [irrep1, irrep2]
                    character_table.append(_cal_irrep_trace(irreps, symprec))
                    quantum_number.append(
                        ((k1, tmp_m1, tmp_piH), (k2, tmp_m2))
                    )
                    index.append(ind)
                    ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_4(
    a: float,
    n: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """T(a)Cnh"""
    # row_labels = [r"$(C_{2n}|1/2)$", r"$C_{n}$", r"$\sigma_{h}$"]
    # column_labels = [r"$_{0}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$", r"$_{\widetilde{k}_{\widetilde{M}}(\widetilde{m})}A_{\widetilde{m}}^{\Pi_{h}}$", r"$_{\widetilde{k}}E_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1) >= symprec:
        judge = False
        message.append("k1 not equal to 0")

    if k2 <= 0 or k2 > np.pi / a:
        judge = False
        message.append("k2 not belong to (0,pi/a]")

    m1 = frac_range(-n, n, left=False)
    m2 = frac_range(-n, n, left=False)
    m3 = frac_range(-n / 2, n / 2, left=False)
    m4 = frac_range(-n / 2, n / 2, left=False)
    piH = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_m3 in m3:
                    for tmp_m4 in m4:
                        for tmp_piH in piH:
                            if k3 <= 2 * np.pi * tmp_m3 / n / a or k3 >= (
                                2 * np.pi / a + 2 * np.pi * tmp_m3 / n / a
                            ):
                                continue
                            if k4 <= 2 * np.pi * tmp_m4 / n / a or k4 >= (
                                2 * np.pi / a + 2 * np.pi * tmp_m4 / n / a
                            ):
                                continue

                            irrep1 = np.round(
                                [
                                    np.exp(1j * tmp_m1 * np.pi / n),
                                    np.exp(1j * tmp_m1 * 2 * np.pi / n),
                                    tmp_piH,
                                ],
                                round_symprec,
                            )
                            irrep2 = np.round(
                                [
                                    [
                                        [
                                            np.exp(
                                                1j
                                                * (
                                                    tmp_m2 * np.pi / n
                                                    + k2 * a / 2
                                                )
                                            ),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.exp(
                                                1j
                                                * (
                                                    tmp_m2 * np.pi / n
                                                    - k2 * a / 2
                                                )
                                            ),
                                        ],
                                    ],
                                    [
                                        [
                                            np.exp(
                                                1j * tmp_m2 * 2 * np.pi / n
                                            ),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.exp(
                                                1j * tmp_m2 * 2 * np.pi / n
                                            ),
                                        ],
                                    ],
                                    [[0, 1], [1, 0]],
                                ],
                                round_symprec,
                            )
                            irrep3 = np.round(
                                [
                                    np.exp(1j * k3 * a / 2),
                                    np.exp(1j * tmp_m3 * 2 * np.pi / n),
                                    tmp_piH,
                                ],
                                round_symprec,
                            )
                            irrep4 = np.round(
                                [
                                    [
                                        [np.exp(1j * k4 * a / 2), 0],
                                        [
                                            0,
                                            np.exp(
                                                1j
                                                * (
                                                    tmp_m4 * 2 * np.pi / n
                                                    - k4 * a / 2
                                                )
                                            ),
                                        ],
                                    ],
                                    [
                                        [
                                            np.exp(
                                                1j * tmp_m4 * 2 * np.pi / n
                                            ),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.exp(
                                                1j * tmp_m4 * 2 * np.pi / n
                                            ),
                                        ],
                                    ],
                                    [[0, 1], [1, 0]],
                                ],
                                round_symprec,
                            )
                            irreps = [irrep1, irrep2, irrep3, irrep4]
                            character_table.append(
                                _cal_irrep_trace(irreps, symprec)
                            )
                            quantum_number.append(
                                (
                                    (k1, tmp_m1, tmp_piH),
                                    (k2, tmp_m2),
                                    (k3, tmp_m3),
                                    (k4, tmp_m4),
                                )
                            )
                            index.append(ind)
                            ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


# Todo: unfinish
def line_group_5(
    q: int,
    r: int,
    a: float,
    f: float,
    n: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """TQ(f)Dn"""
    # row_labels = [r"$(C_{2n}|1/2)$", r"$C_{n}$", r"$\sigma_{h}$"]
    # column_labels = [r"$_{0}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$", r"$_{\widetilde{k}_{\widetilde{M}}(\widetilde{m})}A_{\widetilde{m}}^{\Pi_{h}}$", r"$_{\widetilde{k}}E_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1) >= symprec:
        judge = False
        message.append("k1 not equal to 0")

    if k2 <= 0 or k2 > np.pi / a:
        judge = False
        message.append("k2 not belong to (0,pi/a]")

    m1 = [0, q / 2]
    m2 = []
    m3 = [0, n / 2]
    m4 = frac_range(-n / 2, n / 2, left=False)
    sigmaU = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_m3 in m3:
                    for tmp_m4 in m4:
                        for tmp_sigmah in sigmah:
                            if k3 <= 2 * np.pi * tmp_m3 / n / a or k3 >= (
                                2 * np.pi / a + 2 * np.pi * tmp_m3 / n / a
                            ):
                                continue
                            if k4 <= 2 * np.pi * tmp_m4 / n / a or k4 >= (
                                2 * np.pi / a + 2 * np.pi * tmp_m4 / n / a
                            ):
                                continue

                            irrep1 = np.round(
                                [
                                    np.exp(1j * tmp_m1 * np.pi / n),
                                    np.exp(1j * tmp_m1 * 2 * np.pi / n),
                                    tmp_sigmah,
                                ],
                                round_symprec,
                            )
                            irrep2 = np.round(
                                [
                                    [
                                        [
                                            np.exp(
                                                1j
                                                * (
                                                    tmp_m2 * np.pi / n
                                                    + k2 * a / 2
                                                )
                                            ),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.exp(
                                                1j
                                                * (
                                                    tmp_m2 * np.pi / n
                                                    - k2 * a / 2
                                                )
                                            ),
                                        ],
                                    ],
                                    [
                                        [
                                            np.exp(
                                                1j * tmp_m2 * 2 * np.pi / n
                                            ),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.exp(
                                                1j * tmp_m2 * 2 * np.pi / n
                                            ),
                                        ],
                                    ],
                                    [[0, 1], [1, 0]],
                                ],
                                round_symprec,
                            )
                            irrep3 = np.round(
                                [
                                    np.exp(1j * k3 * a / 2),
                                    np.exp(1j * tmp_m3 * 2 * np.pi / n),
                                    tmp_sigmah,
                                ],
                                round_symprec,
                            )
                            irrep4 = np.round(
                                [
                                    [
                                        [np.exp(1j * k4 * a / 2), 0],
                                        [
                                            0,
                                            np.exp(
                                                1j
                                                * (
                                                    tmp_m4 * 2 * np.pi / n
                                                    - k4 * a / 2
                                                )
                                            ),
                                        ],
                                    ],
                                    [
                                        [
                                            np.exp(
                                                1j * tmp_m4 * 2 * np.pi / n
                                            ),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.exp(
                                                1j * tmp_m4 * 2 * np.pi / n
                                            ),
                                        ],
                                    ],
                                    [[0, 1], [1, 0]],
                                ],
                                round_symprec,
                            )
                            irreps = [irrep1, irrep2, irrep3, irrep4]
                            character_table.append(
                                _cal_irrep_trace(irreps, symprec)
                            )
                            quantum_number.append(
                                (
                                    (k1, tmp_m1, tmp_sigmah),
                                    (k2, tmp_m2),
                                    (k3, tmp_m3),
                                    (k4, tmp_m4),
                                )
                            )
                            index.append(ind)
                            ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_6(
    a: float,
    n: int,
    k1: float,
    k2: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """T(a)Cnv"""
    # label for line group family
    # row_labels = [r"$(C_{Q}|f)$", r"$C_{n}$"]
    # column_labels = [r"$_{k}A_{m}$", r"$_{\widetilde{k}}A_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if k1 <= -np.pi / a or k1 > np.pi / a:
        judge = False
        message.append("k1 not belong to (-pi/a,pi/a]")

    if k2 <= -np.pi / a or k2 > np.pi / a:
        judge = False
        message.append("k2 not belong to (-pi/f,pi/f]")

    m1 = [0, n / 2]
    m2 = frac_range(0, n / 2, left=False, right=False)
    piV = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_piV in piV:
                    irrep1 = np.round(
                        [
                            np.exp(1j * k1 * a),
                            np.exp(1j * tmp_m1 * 2 * np.pi / n),
                            tmp_piV,
                        ],
                        round_symprec,
                    )
                    irrep2 = np.round(
                        [
                            [
                                [np.exp(1j * k2 * a), 0],
                                [0, np.exp(1j * k2 * a)],
                            ],
                            [
                                [np.exp(1j * tmp_m2 * 2 * np.pi / n), 0],
                                [0, np.exp(-1j * tmp_m2 * 2 * np.pi / n)],
                            ],
                            [[0, 1], [1, 0]],
                        ],
                        round_symprec,
                    )
                    irreps = [irrep1, irrep2]
                    character_table.append(_cal_irrep_trace(irreps, symprec))
                    quantum_number.append(
                        ((k1, tmp_m1, tmp_piV), (k2, tmp_m2))
                    )
                    index.append(ind)
                    ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_7(
    a: float,
    n: int,
    k1: float,
    k2: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """T'(a/2)Cn"""
    # label for line group family
    # row_labels = [r"$(C_{Q}|f)$", r"$C_{n}$"]
    # column_labels = [r"$_{k}A_{m}$", r"$_{\widetilde{k}}A_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if k1 <= -np.pi / a or k1 > np.pi / a:
        judge = False
        message.append("k1 not belong to (-pi/a,pi/a]")

    if k2 <= -np.pi / a or k2 > np.pi / a:
        judge = False
        message.append("k2 not belong to (-pi/f,pi/f]")

    m1 = [0, n / 2]
    m2 = frac_range(0, n / 2, left=False, right=False)
    piV = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_piV in piV:
                    irrep1 = np.round(
                        [
                            tmp_piV * np.exp(1j * k1 * a),
                            np.exp(1j * tmp_m1 * 2 * np.pi / n),
                        ],
                        round_symprec,
                    )
                    irrep2 = np.round(
                        [
                            [
                                [0, np.exp(1j * k2 * a / 2)],
                                [np.exp(1j * k2 * a / 2), 0],
                            ],
                            [
                                [np.exp(1j * tmp_m2 * 2 * np.pi / n), 0],
                                [0, np.exp(-1j * tmp_m2 * 2 * np.pi / n)],
                            ],
                        ],
                        round_symprec,
                    )
                    irreps = [irrep1, irrep2]
                    character_table.append(_cal_irrep_trace(irreps, symprec))
                    quantum_number.append(
                        ((k1, tmp_m1, tmp_piV), (k2, tmp_m2))
                    )
                    index.append(ind)
                    ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


# Todo
def line_group_8():
    pass


# Todo: m must be integer?
def line_group_9(
    a: float,
    n: int,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,
    symprec: float = 1e-4,
    round_symprec: int = 3,
) -> CharacterDataset:
    """T(a)Dnd"""
    # label for line group family
    # row_labels = [r"$(C_{Q}|f)$", r"$C_{n}$"]
    # column_labels = [r"$_{k}A_{m}$", r"$_{\widetilde{k}}A_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if k1 != 0 and k1 != np.pi / a:
        judge = False
        message.append("k1 not belong to 0 or pi/a")

    if k2 != 0 and k2 != np.pi / a:
        judge = False
        message.append("k2 not belong to 0 or pi/a")

    if k3 != 0 and k3 != np.pi / a:
        judge = False
        message.append("k3 not belong to 0 or pi/a")

    if k4 <=0 or k4 >= np.pi / a:
        judge = False
        message.append("k4 not belong to (0,pi/a)")

    if k5 <=0 or k5 >= np.pi / a:
        judge = False
        message.append("k5 not belong to (0,pi/a)")


    m1 = [0]
    m2 = frac_range(0, n / 2, left=False, right=False)
    m3 = [n/2]
    m4 = [0, n/2]
    m5 = frac_range(0, n / 2, left=False, right=False)
    sigmaU = [-1, 1]
    sigmaV = [-1, 1]
    sigmaH = [-1, 1]
    piV = [-1, 1]

    if judge:
        ind = 0
        quantum_number = []
        character_table = []
        index = []
        for tmp_m1 in m1:
            for tmp_m2 in m2:
                for tmp_m3 in m3:
                    for tmp_m4 in m4:
                        for tmp_m5 in m5:
                            for tmp_piV in piV:
                                for tmp_sigmaH in sigmaH:
                                    for tmp_sigmaU in sigmaU:
                                        for tmp_sigmaV in sigmaV:
                                            irrep1 = np.round(
                                                [
                                                    np.exp(1j * k1 * a),
                                                    1,
                                                    tmp_sigmaU,
                                                    tmp_sigmaV
                                                ],
                                                round_symprec,
                                            )
                                            # set_trace()
                                            irrep2 = np.round(
                                                [
                                                    [
                                                        [np.exp(1j*k2*a), 0],
                                                        [0, np.exp(1j*k2*a)],
                                                    ],
                                                    [
                                                        [np.exp(1j * tmp_m2 * 2 * np.pi / n), 0],
                                                        [0, np.exp(-1j * tmp_m2 * 2 * np.pi / n)],
                                                    ],
                                                    tmp_sigmaH*[
                                                        [0, np.exp(1j * tmp_m2 * np.pi / n)],
                                                        [np.exp(-1j * tmp_m2 * np.pi / n), 0],
                                                    ],
                                                    [
                                                        [0, 1],
                                                        [1, 0],
                                                    ],
                                                ],
                                                round_symprec,
                                            )
                                            irrep3 = np.round(
                                                [
                                                    np.exp(1j*k3*a)*[
                                                        [1, 0],
                                                        [0, 1],
                                                    ],
                                                    [
                                                        [-1, 0],
                                                        [0, -1],
                                                    ],
                                                    [
                                                        [0, 1],
                                                        [1, 0],
                                                    ],
                                                    [
                                                        [1, 0],
                                                        [0, -1],
                                                    ],
                                                ],
                                                round_symprec,
                                            )
                                            irrep4 = np.round(
                                                [
                                                    [
                                                        [np.exp(1j*k2*a), 0],
                                                        [0, np.exp(-1j*k2*a)],
                                                    ],
                                                    np.exp(1j * tmp_m4 * 2 * np.pi / n)*[
                                                        [1, 0],
                                                        [0, 1],
                                                    ],
                                                    [
                                                        [0, 1],
                                                        [1, 0],
                                                    ],
                                                    tmp_sigmaV*[
                                                        [1, 0],
                                                        [0, np.exp(1j * tmp_m4 * 2 * np.pi / n)],
                                                    ],
                                                ],
                                                round_symprec,
                                            )
                                            irrep5 = np.round(
                                                [
                                                    [
                                                        [np.exp(1j*k5*a), 0, 0, 0],
                                                        [0, np.exp(1j*k5*a), 0, 0],
                                                        [0, 0, np.exp(-1j*k5*a), 0],
                                                        [0, 0, 0, np.exp(-1j*k5*a)],
                                                    ],
                                                    [
                                                        [np.exp(1j*tmp_m5*2*np.pi/n), 0, 0, 0],
                                                        [0, np.exp(-1j*tmp_m5*2*np.pi/n), 0, 0],
                                                        [0, 0, np.exp(-1j*tmp_m5*2*np.pi/n), 0],
                                                        [0, 0, 0, np.exp(1j*tmp_m5*2*np.pi/n)],
                                                    ],
                                                    [
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1],
                                                        [1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                    ],
                                                    [
                                                        [0, 1, 0, 0],
                                                        [1, 0, 0, 0],
                                                        [0, 0, 0, np.exp(1j*tmp_m5*2*np.pi/n)],
                                                        [0, 0, np.exp(-1j*tmp_m5*2*np.pi/n), 0],
                                                    ],
                                                ],
                                                round_symprec,
                                            )

                                            irreps = [irrep1, irrep2, irrep3, irrep4, irrep5]
                                            character_table.append(_cal_irrep_trace(irreps, symprec))
                                            quantum_number.append(
                                                ((k1, tmp_m1, tmp_piV), (k2, tmp_m2))
                                            )
                                            index.append(ind)
                                            ind = ind + 1
        return CharacterDataset(index, quantum_number, character_table)
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)




def plot_character_table(character, row, column):
    plt.figure(dpi=200)
    # fig,ax = plt.subplots(1,2)
    # set_trace()
    plt.axis("off")
    plt.table(
        cellText=character,
        rowLabels=column,
        colLabels=row,
        colWidths=[0.2 for x in row],
        # colColours=colColors,
        # rowColours=rowColours,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    # ax[1].axis('off')
    # ax[1].text("123456")
    # plt.savefig("fig2.png", dpi=400)
    plt.show()


def main():
    # # line 1
    # q, r, f, n = 3, 1, 3, 5
    # Q = q/r
    # a = Q*f
    # k1, k2 = 0, 0

    # # line 2
    # a = 3
    # n = 6
    # k1 = np.pi/a
    # k2 = np.pi/3

    # # line 3
    # a = 3
    # n = 6
    # k1 = np.pi/a
    # k2 = np.pi/3

    # line 4
    a = 3
    n = 6
    k1 = 0
    k2 = np.pi / a
    k3 = 0
    k4 = np.pi / 2 / a
    k5 = np.pi / 3 / a

    # dataset = line_group_1(q, r, a, f, n, k1, k2)
    # dataset = line_group_2(a, n, k1, k2)
    dataset = line_group_9(a, n, k1, k2,k3,k4, k5)

    set_trace()
    # character_tabel, row_labels, column_labels = line_group_4(a, n, k1, m1, k2, m2, k3, m3, k4, m4, sigmah)


if __name__ == "__main__":
    main()
