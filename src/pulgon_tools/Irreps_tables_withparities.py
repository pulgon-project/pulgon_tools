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


import itertools
import logging
from fractions import Fraction
from typing import List, Tuple

import numpy as np
import sympy
from sympy import symbols


def sym_inverse_eye(n: int) -> sympy.Matrix:
    A = sympy.zeros(n)
    for ii in range(n):
        A[ii, n - ii - 1] = 1
    return A


def _value_fc(
    fc: list,
    subs: dict,
    order: List[List[int]],
    *,
    left_multiply: bool = True,
) -> list:
    res = []
    for tmp_order in order:
        for jj, tmp in enumerate(tmp_order):
            if jj == 0:
                tmp0 = fc[tmp]
            elif left_multiply:
                tmp0 = fc[tmp] * tmp0
            else:
                tmp0 = tmp0 * fc[tmp]
        res.append(tmp0.evalf(subs=subs))
    return res


def _as_complex_array(values: list) -> np.ndarray:
    return np.array(values).astype(np.complex128)


def _m_values_signed(nrot: int) -> list:
    return list(range(-int(nrot / 2) + 1, int(nrot / 2) + 1))


def _m_values_full(nrot: int) -> list:
    return list(range(-int(nrot) + 1, int(nrot) + 1))


def _m_values_nonnegative(nrot: int) -> list:
    return list(range(0, int(np.floor(nrot / 2)) + 1))


def _m_values_screw(q_num: int) -> list:
    start = int(np.floor(-q_num / 2.0)) + 1
    stop = int(np.floor(q_num / 2.0)) + 1
    return list(range(start, stop))


def _is_boundary_m(m_value: float, nrot: int, symprec: float) -> bool:
    return np.isclose(m_value, 0, atol=symprec) or np.isclose(
        m_value, nrot / 2, atol=symprec
    )


def _phase2(m1, n):
    return sympy.exp(1j * m1 * 2 * sympy.pi / n)


def _swap_with_angle(m1, angle):
    return sympy.Matrix(
        [
            [0, sympy.exp(-1j * 2 * m1 * angle)],
            [sympy.exp(1j * 2 * m1 * angle), 0],
        ]
    )


def _swap() -> sympy.Matrix:
    return sympy.Matrix([[0, 1], [1, 0]])


def line_group_sympy_withparities(
    DictParams: dict, symprec: float = 1e-6
) -> Tuple[list, list, list]:
    family = DictParams["family"]

    if family == 1:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        order = DictParams["order"]
        q_screw = DictParams.get("Q_screw")
        f_screw = DictParams.get("f_screw")
        if q_screw is None or f_screw is None:
            raise KeyError(
                "Family 1 requires 'Q_screw' and 'f_screw' in DictParams."
            )

        q_frac = Fraction(str(q_screw)).limit_denominator(1000)
        q_num = DictParams.get("Q_num", q_frac.numerator)

        k1, m1, n, Q, f = symbols("k1 m1 n Q f")
        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * (k1 * f + m1 * 2 * sympy.pi / Q)),
                sympy.exp(1j * m1 * 2 * sympy.pi / n),
            ]
        )

        m1_value = _m_values_screw(q_num)

        paras_symbol = [k1, m1]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product([qpoint], m1_value):
            res = _value_fc(
                func0,
                {k1: tmp_k1, m1: tmp_m1, n: nrot, Q: q_screw, f: f_screw},
                order,
            )
            characters.append(_as_complex_array(res))
            paras_values.append([tmp_k1, tmp_m1])

    elif family == 2:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        k1, m1, n, piH = symbols("k1 m1 n piH")
        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * k1 * a),
                piH * sympy.exp(1j * m1 * sympy.pi / n),
            ]
        )
        func1 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * k1 * a),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(-1j * k1 * a),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        0,
                        sympy.exp(1j * m1 * 2 * sympy.pi / n),
                    ],
                    [
                        1,
                        0,
                    ],
                ]
            ),
        ]
        paras_symbol = [k1, m1, piH]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_signed(nrot)
        ):
            if np.isclose(tmp_k1, 0, atol=symprec) or np.isclose(
                tmp_k1, np.pi / a, atol=symprec
            ):
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func0,
                        {k1: tmp_k1, m1: tmp_m1, n: nrot, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piH])
            else:
                tmp_piH = 0
                res = _value_fc(
                    func1,
                    {k1: tmp_k1, m1: tmp_m1, n: nrot, piH: tmp_piH},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, tmp_piH])

    elif family == 3:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        k1, m1, n, piH = symbols("k1 m1 n piH")
        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * k1 * a),
                _phase2(m1, n),
                piH,
            ]
        )
        func1 = [
            sympy.eye(2),
            sympy.Matrix(
                [
                    [sympy.exp(1j * k1 * a), 0],
                    [0, sympy.exp(-1j * k1 * a)],
                ]
            ),
            sympy.exp(1j * m1 * 2 * sympy.pi / n) * sympy.eye(2),
            _swap(),
        ]

        paras_symbol = [k1, m1, piH]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_signed(nrot)
        ):
            if np.isclose(tmp_k1, 0, atol=symprec) or np.isclose(
                tmp_k1, np.pi / a, atol=symprec
            ):
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func0,
                        {k1: tmp_k1, m1: tmp_m1, n: nrot, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piH])
            else:
                res = _value_fc(
                    func1,
                    {k1: tmp_k1, m1: tmp_m1, n: nrot, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0])

    elif family == 4:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        k1, m1, n, piH = symbols("k1 m1 n piH")
        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * m1 * sympy.pi / n),
                sympy.exp(1j * m1 * 2 * sympy.pi / n),
                piH,
            ]
        )
        func1 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * (m1 * sympy.pi / n + k1 * a / 2)),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * (m1 * sympy.pi / n - k1 * a / 2)),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * m1 * 2 * sympy.pi / n),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * m1 * 2 * sympy.pi / n),
                    ],
                ]
            ),
            sympy.Matrix([[0, 1], [1, 0]]),
        ]
        paras_symbol = [k1, m1, piH]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_full(nrot)
        ):
            if np.isclose(tmp_k1, 0, atol=symprec):
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func0,
                        {k1: tmp_k1, m1: tmp_m1, n: nrot, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piH])
            else:
                tmp_piH = 0
                res = _value_fc(
                    func1,
                    {k1: tmp_k1, m1: tmp_m1, n: nrot, piH: tmp_piH},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, tmp_piH])

    elif family == 5:
        # Family 5 (T_Q D_n) irrep table.
        #
        # The original tabulation was inconsistent with the |G|-sum rule
        # (sum d_mu^2 != |G|) at every q-point.  Two corrections,
        # mirroring the family-13 derivation, restore consistency:
        #
        # 1. The C_n axial rotation contributes a phase
        #    exp(i*m*2*pi/n_axial); LineGroupAnalyzer reports
        #    nrot = 2*n_axial for the D_n point group, so the formula
        #    becomes exp(i*m*4*pi/nrot) (matching family 13's func1).
        # 2. At Gamma the (+m, -m) pair is already encoded in the 2D
        #    representation, so the m-range is reduced to
        #    [0, floor(Q/2)] to avoid double counting.  Self-paired m
        #    (m=0 and, when Q is even, m=Q/2) split into a 1D piU=+/-1
        #    branch.  Sum d^2 = |G| is recovered.
        # 3. At non-zero q the (+k, -k) sectors must be carried
        #    explicitly.  Family 5 has no horizontal mirror, so a
        #    family-13-style 4D representation on
        #    (|+m,+k>, |-m,+k>, |+m,-k>, |-m,-k>) is reducible under
        #    the lone U_x generator.  Instead, every m in [0, Q-1]
        #    is given its own 2D representation:
        #      * self-paired m (m=0 or m=Q/2): the 2D basis is
        #        (|m,+k>, |m,-k>);  U_x swaps the two k-sectors;
        #      * generic m: the 2D basis is (|+m,+k>, |-m,-k>);
        #        U_x simultaneously flips m and k.
        #    The little-group filter in the projector code selects the
        #    +k sector for each m and the Q distinct +k characters
        #    cover the full Hilbert space at the chosen k.
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        order = DictParams["order"]
        q_screw = DictParams.get("Q_screw")
        f_screw = DictParams.get("f_screw")
        if q_screw is None or f_screw is None:
            raise KeyError(
                "Family 5 requires 'Q_screw' and 'f_screw' in DictParams."
            )
        q_frac = Fraction(str(q_screw)).limit_denominator(1000)
        q_num = DictParams.get("Q_num", q_frac.numerator)

        k1, m1, n, Q, f, piU, alphaU = symbols("k1 m1 n Q f piU alphaU")
        screw_phase = sympy.exp(1j * (k1 * f + m1 * 2 * sympy.pi / Q))
        screw_phase_mk = sympy.exp(1j * (-k1 * f + m1 * 2 * sympy.pi / Q))
        rot_phase = sympy.exp(1j * 4 * m1 * sympy.pi / n)

        # 1D rep at Gamma for self-paired m (m=0 or m=Q/2 when Q even).
        func0 = sympy.Matrix([1, screw_phase, rot_phase, piU])
        # 2D rep at Gamma for generic 0 < m < Q/2 on (|+m>, |-m>),
        # and at non-zero q for generic m on (|+m,+k>, |-m,-k>).
        func1 = [
            sympy.eye(2),
            sympy.Matrix([[screw_phase, 0], [0, 1 / screw_phase]]),
            sympy.Matrix([[rot_phase, 0], [0, 1 / rot_phase]]),
            _swap_with_angle(m1, alphaU),
        ]
        # 2D rep at non-zero q for self-paired m on (|m,+k>, |m,-k>).
        # The screw matrix carries opposite Bloch phases on the two
        # k-sectors; the axial rotation is a scalar; U_x swaps them.
        func2 = [
            sympy.eye(2),
            sympy.Matrix([[screw_phase, 0], [0, screw_phase_mk]]),
            sympy.Matrix([[rot_phase, 0], [0, rot_phase]]),
            _swap_with_angle(m1, alphaU),
        ]

        tmp_alphaU = DictParams.get("alphaU", 0.0)
        if np.isclose(qpoint, 0, atol=symprec):
            m1_value = list(range(0, int(q_num) // 2 + 1))
        else:
            m1_value = list(range(0, int(q_num)))
        paras_symbol = [k1, m1, piU]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product([qpoint], m1_value):
            is_special_m = np.isclose(tmp_m1, 0, atol=symprec) or (
                int(q_num) % 2 == 0
                and np.isclose(tmp_m1, q_num / 2, atol=symprec)
            )
            base_subs = {
                k1: tmp_k1,
                m1: tmp_m1,
                n: nrot,
                Q: q_screw,
                f: f_screw,
                alphaU: tmp_alphaU,
            }

            def eval_fc(fc, tmp_piU=0):
                return _value_fc(fc, {**base_subs, piU: tmp_piU}, order)

            if np.isclose(tmp_k1, 0, atol=symprec):
                if is_special_m:
                    for tmp_piU in [-1, 1]:
                        res = eval_fc(func0, tmp_piU)
                        characters.append(_as_complex_array(res))
                        paras_values.append([tmp_k1, tmp_m1, tmp_piU])
                else:
                    res = eval_fc(func1, 0)
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0])
            else:
                if is_special_m:
                    res = eval_fc(func2, 0)
                else:
                    res = eval_fc(func1, 0)
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0])

    elif family == 6:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        k1, m1, n, piV = symbols("k1 m1 n piV")

        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * k1 * a),
                sympy.exp(1j * m1 * 2 * sympy.pi / n),
                piV,
            ]
        )
        func1 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * k1 * a),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * k1 * a),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * m1 * 2 * sympy.pi / n),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(-1j * m1 * 2 * sympy.pi / n),
                    ],
                ]
            ),
            sympy.Matrix([[0, 1], [1, 0]]),
        ]
        paras_symbol = [k1, m1, piV]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            if _is_boundary_m(tmp_m1, nrot, symprec):
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func0,
                        {k1: tmp_k1, m1: tmp_m1, n: nrot, piV: tmp_piV},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piV])
            else:
                tmp_piV = 0
                res = _value_fc(
                    func1,
                    {k1: tmp_k1, m1: tmp_m1, n: nrot, piV: tmp_piV},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, tmp_piV])

    elif family == 7:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        k1, m1, n, piV = symbols("k1 m1 n piV")
        glide_phase = sympy.exp(1j * k1 * a / 2)
        rot_phase = _phase2(m1, n)
        func0 = sympy.Matrix([1, piV * glide_phase, rot_phase])
        func1 = [
            sympy.eye(2),
            sympy.Matrix([[0, glide_phase], [glide_phase, 0]]),
            sympy.Matrix([[rot_phase, 0], [0, 1 / rot_phase]]),
        ]

        paras_symbol = [k1, m1, piV]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            if _is_boundary_m(tmp_m1, nrot, symprec):
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func0,
                        {k1: tmp_k1, m1: tmp_m1, n: nrot, piV: tmp_piV},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piV])
            else:
                res = _value_fc(
                    func1,
                    {k1: tmp_k1, m1: tmp_m1, n: nrot, piV: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0])

    elif family == 8:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]

        k1, m1, n, piV = symbols("k1 m1 n piV")
        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * (k1 * a / 2 + m1 * sympy.pi / n)),
                1,
                piV,
            ]
        )

        func1 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * (k1 * a / 2 + m1 * sympy.pi / n)),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * (k1 * a / 2 - m1 * sympy.pi / n)),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * m1 * 2 * sympy.pi / n),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(-1j * m1 * 2 * sympy.pi / n),
                    ],
                ]
            ),
            sympy.Matrix([[0, 1], [1, 0]]),
        ]

        paras_symbol = [k1, m1, piV]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product([qpoint], range(0, nrot + 1)):
            if np.isclose(tmp_m1, 0, atol=symprec) or np.isclose(
                tmp_m1, nrot, atol=symprec
            ):
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func0,
                        {k1: tmp_k1, m1: tmp_m1, n: nrot, piV: tmp_piV},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piV])
            else:
                tmp_piV = 0
                res = _value_fc(
                    func1,
                    {k1: tmp_k1, m1: tmp_m1, n: nrot, piV: tmp_piV},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, tmp_piV])

    elif family == 9:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        n, k1, m1, piU, piV, piH, alphaU, betaS = symbols(
            "n k1 m1 piU piV piH alphaU betaS"
        )
        trans_k = sympy.exp(1j * k1 * a)
        rot_m = _phase2(m1, n)
        tmp_alphaU = DictParams.get("alphaU", 0.0)
        tmp_betaS = DictParams.get("betaS", 0.0)

        func0 = sympy.Matrix([1, trans_k, rot_m, piU, piV])
        func1 = [
            sympy.eye(2),
            trans_k * sympy.eye(2),
            sympy.Matrix([[rot_m, 0], [0, 1 / rot_m]]),
            piH * _swap_with_angle(m1, alphaU),
            _swap_with_angle(m1, betaS),
        ]
        func2 = [
            sympy.eye(2),
            sympy.Matrix([[trans_k, 0], [0, 1 / trans_k]]),
            rot_m * sympy.eye(2),
            _swap(),
            piV * sympy.eye(2),
        ]
        func4 = [
            sympy.eye(4),
            sympy.diag(trans_k, trans_k, 1 / trans_k, 1 / trans_k),
            sympy.diag(rot_m, 1 / rot_m, 1 / rot_m, rot_m),
            sympy.Matrix(
                [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
            ),
            sympy.Matrix(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
            ),
        ]
        paras_symbol = [k1, m1, piU, piV, piH]

        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            base_subs = {
                k1: tmp_k1,
                m1: tmp_m1,
                n: nrot,
                alphaU: tmp_alphaU,
                betaS: tmp_betaS,
            }
            is_k_edge = np.isclose(tmp_k1, 0, atol=symprec) or np.isclose(
                tmp_k1, np.pi / a, atol=symprec
            )
            is_m_edge = _is_boundary_m(tmp_m1, nrot, symprec)
            if is_k_edge and is_m_edge:
                for tmp_piU, tmp_piV, tmp_piH in itertools.product(
                    [-1, 1], [-1, 1], [-1, 1]
                ):
                    res = _value_fc(
                        func0,
                        {
                            **base_subs,
                            piU: tmp_piU,
                            piV: tmp_piV,
                            piH: tmp_piH,
                        },
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append(
                        [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                    )
            elif is_k_edge:
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func1,
                        {**base_subs, piU: 0, piV: 0, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, 0, tmp_piH])
            elif is_m_edge:
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func2,
                        {**base_subs, piU: 0, piV: tmp_piV, piH: 0},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, tmp_piV, 0])
            else:
                res = _value_fc(
                    func4,
                    {**base_subs, piU: 0, piV: 0, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0, 0, 0])

    elif family == 11:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        n, k1, m1, piU, piV, piH, alphaU, betaS = symbols(
            "n k1 m1 piU piV piH alphaU betaS"
        )
        trans_k = sympy.exp(1j * k1 * a)
        rot_m = _phase2(m1, n)
        tmp_alphaU = DictParams.get("alphaU", 0.0)
        tmp_betaS = DictParams.get("betaS", 0.0)

        func0 = sympy.Matrix([1, trans_k, rot_m, piV, piH])
        func1 = [
            sympy.eye(2),
            trans_k * sympy.eye(2),
            sympy.Matrix([[rot_m, 0], [0, 1 / rot_m]]),
            _swap_with_angle(m1, betaS),
            piH * sympy.eye(2),
        ]
        func2 = [
            sympy.eye(2),
            sympy.Matrix([[trans_k, 0], [0, 1 / trans_k]]),
            rot_m * sympy.eye(2),
            piV * sympy.eye(2),
            _swap(),
        ]
        func4 = [
            sympy.eye(4),
            sympy.diag(trans_k, trans_k, 1 / trans_k, 1 / trans_k),
            sympy.diag(rot_m, 1 / rot_m, rot_m, 1 / rot_m),
            sympy.Matrix(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
            ),
            sympy.Matrix(
                [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
            ),
        ]
        paras_symbol = [k1, m1, piU, piV, piH]

        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            base_subs = {
                k1: tmp_k1,
                m1: tmp_m1,
                n: nrot,
                alphaU: tmp_alphaU,
                betaS: tmp_betaS,
            }
            is_k_edge = np.isclose(tmp_k1, 0, atol=symprec) or np.isclose(
                tmp_k1, np.pi / a, atol=symprec
            )
            is_m_edge = _is_boundary_m(tmp_m1, nrot, symprec)
            if is_k_edge and is_m_edge:
                for tmp_piU, tmp_piV, tmp_piH in itertools.product(
                    [-1, 1], [-1, 1], [-1, 1]
                ):
                    res = _value_fc(
                        func0,
                        {
                            **base_subs,
                            piU: tmp_piU,
                            piV: tmp_piV,
                            piH: tmp_piH,
                        },
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append(
                        [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                    )
            elif is_k_edge:
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func1,
                        {**base_subs, piU: 0, piV: 0, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, 0, tmp_piH])
            elif is_m_edge:
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func2,
                        {**base_subs, piU: 0, piV: tmp_piV, piH: 0},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, tmp_piV, 0])
            else:
                res = _value_fc(
                    func4,
                    {**base_subs, piU: 0, piV: 0, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0, 0, 0])

    elif family == 10:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        n, k1, m1, piU, piV, piH = symbols("n k1 m1 piU piV piH")
        glide_k = sympy.exp(1j * k1 * a / 2)
        s_phase = sympy.exp(1j * m1 * sympy.pi / n)
        rot_m = _phase2(m1, n)
        func0 = sympy.Matrix([1, piV * glide_k, piU * piV * s_phase])
        func1 = [
            sympy.eye(2),
            glide_k * _swap(),
            piH * sympy.Matrix([[s_phase, 0], [0, 1 / s_phase]]),
        ]
        func2 = [
            sympy.eye(2),
            piV * sympy.Matrix([[glide_k, 0], [0, 1 / glide_k]]),
            sympy.Matrix([[0, s_phase], [1 / s_phase, 0]]),
        ]
        func3 = [
            sympy.eye(2),
            sympy.Matrix([[glide_k, 0], [0, -glide_k]]),
            sympy.Matrix([[0, -sympy.exp(1j * k1 * a)], [1, 0]]),
        ]
        func4 = [
            sympy.eye(4),
            sympy.Matrix(
                [
                    [0, glide_k, 0, 0],
                    [glide_k, 0, 0, 0],
                    [0, 0, 0, 1 / glide_k],
                    [0, 0, 1 / glide_k, 0],
                ]
            ),
            sympy.Matrix(
                [
                    [0, 0, s_phase, 0],
                    [0, 0, 0, 1 / s_phase],
                    [s_phase, 0, 0, 0],
                    [0, 1 / s_phase, 0, 0],
                ]
            ),
        ]

        paras_symbol = [k1, m1, piU, piV, piH]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            base_subs = {k1: tmp_k1, m1: tmp_m1, n: nrot}
            is_gamma = np.isclose(tmp_k1, 0, atol=symprec)
            is_pi = np.isclose(tmp_k1, np.pi / a, atol=symprec)
            is_m_edge = _is_boundary_m(tmp_m1, nrot, symprec)
            if (is_gamma and np.isclose(tmp_m1, 0, atol=symprec)) or (
                is_pi and np.isclose(tmp_m1, nrot / 2, atol=symprec)
            ):
                for tmp_piU, tmp_piV in itertools.product([-1, 1], [-1, 1]):
                    res = _value_fc(
                        func0,
                        {
                            **base_subs,
                            piU: tmp_piU,
                            piV: tmp_piV,
                            piH: 0,
                        },
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piU, tmp_piV, 0])
            elif (is_gamma or is_pi) and not is_m_edge:
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func1,
                        {**base_subs, piU: 0, piV: 0, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, 0, tmp_piH])
            elif is_m_edge and not (is_gamma or is_pi):
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func2,
                        {**base_subs, piU: 0, piV: tmp_piV, piH: 0},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, tmp_piV, 0])
            elif (is_gamma and np.isclose(tmp_m1, nrot / 2, atol=symprec)) or (
                is_pi and np.isclose(tmp_m1, 0, atol=symprec)
            ):
                res = _value_fc(
                    func3,
                    {**base_subs, piU: 0, piV: 0, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0, 0, 0])
            else:
                res = _value_fc(
                    func4,
                    {**base_subs, piU: 0, piV: 0, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0, 0, 0])

    elif family == 12:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        n, k1, m1, piV, piH = symbols("n k1 m1 piV piH")
        glide_k = sympy.exp(1j * k1 * a / 2)
        rot_m = _phase2(m1, n)
        func0 = sympy.Matrix([1, piV * glide_k, rot_m, piH])
        func1 = [
            sympy.eye(2),
            glide_k * _swap(),
            sympy.Matrix([[rot_m, 0], [0, 1 / rot_m]]),
            piH * sympy.eye(2),
        ]
        func2 = [
            sympy.eye(2),
            piV * sympy.Matrix([[glide_k, 0], [0, 1 / glide_k]]),
            rot_m * sympy.eye(2),
            _swap(),
        ]
        func3 = [
            sympy.eye(2),
            sympy.Matrix([[sympy.I, 0], [0, -sympy.I]]),
            rot_m * sympy.eye(2),
            _swap(),
        ]
        func4 = [
            sympy.eye(4),
            sympy.Matrix(
                [
                    [0, glide_k, 0, 0],
                    [glide_k, 0, 0, 0],
                    [0, 0, 0, 1 / glide_k],
                    [0, 0, 1 / glide_k, 0],
                ]
            ),
            sympy.diag(rot_m, 1 / rot_m, rot_m, 1 / rot_m),
            sympy.Matrix(
                [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
            ),
        ]

        paras_symbol = [k1, m1, piV, piH]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            base_subs = {k1: tmp_k1, m1: tmp_m1, n: nrot}
            is_gamma = np.isclose(tmp_k1, 0, atol=symprec)
            is_pi = np.isclose(tmp_k1, np.pi / a, atol=symprec)
            is_m_edge = _is_boundary_m(tmp_m1, nrot, symprec)
            if is_gamma and is_m_edge:
                for tmp_piV, tmp_piH in itertools.product([-1, 1], [-1, 1]):
                    res = _value_fc(
                        func0,
                        {**base_subs, piV: tmp_piV, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piV, tmp_piH])
            elif (is_gamma or is_pi) and not is_m_edge:
                for tmp_piH in [-1, 1]:
                    res = _value_fc(
                        func1,
                        {**base_subs, piV: 0, piH: tmp_piH},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, 0, tmp_piH])
            elif is_pi and is_m_edge:
                res = _value_fc(
                    func3,
                    {**base_subs, piV: 0, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0, 0])
            elif is_m_edge:
                for tmp_piV in [-1, 1]:
                    res = _value_fc(
                        func2,
                        {**base_subs, piV: tmp_piV, piH: 0},
                        order,
                    )
                    characters.append(_as_complex_array(res))
                    paras_values.append([tmp_k1, tmp_m1, tmp_piV, 0])
            else:
                res = _value_fc(
                    func4,
                    {**base_subs, piV: 0, piH: 0},
                    order,
                )
                characters.append(_as_complex_array(res))
                paras_values.append([tmp_k1, tmp_m1, 0, 0])

    elif family == 13:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]
        n, k1, m1, f, piU, piV, piH, alphaU, betaS = symbols(
            "n k1 m1 f piU piV piH alphaU betaS"
        )

        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * 2 * m1 * sympy.pi / n),
                1,
                piU,
                piV,
            ]
        )
        func1 = [
            sympy.Matrix(
                [
                    [1, 0],
                    [0, 1],
                ]
            ),
            sympy.Matrix(
                [
                    [sympy.exp(1j * 2 * m1 * sympy.pi / n), 0],
                    [0, sympy.exp(-1j * 2 * m1 * sympy.pi / n)],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * 4 * m1 * sympy.pi / n),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(-1j * 4 * m1 * sympy.pi / n),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        0,
                        piH * sympy.exp(-1j * 2 * m1 * alphaU),
                    ],
                    [
                        piH * sympy.exp(1j * 2 * m1 * alphaU),
                        0,
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [0, sympy.exp(-1j * 2 * m1 * betaS)],
                    [sympy.exp(1j * 2 * m1 * betaS), 0],
                ]
            ),
        ]
        func2 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * (2 * m1 * sympy.pi / n + k1 * a / 2)),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * (2 * m1 * sympy.pi / n - k1 * a / 2)),
                    ],
                ]
            ),
            # sympy.Matrix(
            #     [[0, 1], [1, 0]]
            # ),  # Suspecting it's an error in the book of line group
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix([[0, piV], [piV, 0]]),
            sympy.Matrix([[piV, 0], [0, piV]]),
        ]
        func3 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix([[-1, 0], [0, 1]]),
            sympy.Matrix([[-1, 0], [0, -1]]),
            sympy.Matrix([[piU, 0], [0, piU]]),
            sympy.Matrix([[0, 1], [1, 0]]),
        ]
        func4 = [
            sympy.Matrix(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * (2 * m1 * sympy.pi / n + k1 * a / 2)),
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * (-2 * m1 * sympy.pi / n + k1 * a / 2)),
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        sympy.exp(1j * (2 * m1 * sympy.pi / n - k1 * a / 2)),
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        sympy.exp(-1j * (2 * m1 * sympy.pi / n + k1 * a / 2)),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [sympy.exp(1j * 4 * sympy.pi * m1 / n), 0, 0, 0],
                    [0, sympy.exp(-1j * 4 * sympy.pi * m1 / n), 0, 0],
                    [0, 0, sympy.exp(1j * 4 * sympy.pi * m1 / n), 0],
                    [
                        0,
                        0,
                        0,
                        sympy.exp(-1j * 4 * sympy.pi * m1 / n),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [0, 0, 0, sympy.exp(-1j * 2 * m1 * alphaU)],
                    [0, 0, sympy.exp(1j * 2 * m1 * alphaU), 0],
                    [0, sympy.exp(-1j * 2 * m1 * alphaU), 0, 0],
                    [sympy.exp(1j * 2 * m1 * alphaU), 0, 0, 0],
                ]
            ),
            sympy.Matrix(
                [
                    [0, sympy.exp(-1j * 2 * m1 * betaS), 0, 0],
                    [sympy.exp(1j * 2 * m1 * betaS), 0, 0, 0],
                    [0, 0, 0, sympy.exp(-1j * 2 * m1 * betaS)],
                    [0, 0, sympy.exp(1j * 2 * m1 * betaS), 0],
                ]
            ),
        ]

        tmp_alphaU = DictParams.get("alphaU", 0.0)
        tmp_betaS = DictParams.get("betaS", 0.0)

        paras_symbol = [k1, m1, piU, piV, piH]
        characters, paras_values = [], []
        for tmp_k1, tmp_m1 in itertools.product(
            [qpoint], _m_values_nonnegative(nrot)
        ):
            base_subs = {
                k1: tmp_k1,
                m1: tmp_m1,
                n: nrot,
                f: a / 2,
                alphaU: tmp_alphaU,
                betaS: tmp_betaS,
            }

            def eval_fc(fc, tmp_piU=0, tmp_piV=0, tmp_piH=0):
                return _value_fc(
                    fc,
                    {
                        **base_subs,
                        piU: tmp_piU,
                        piV: tmp_piV,
                        piH: tmp_piH,
                    },
                    order,
                    left_multiply=False,
                )

            if np.isclose(tmp_k1, 0, atol=symprec):
                if (
                    np.isclose(np.abs(tmp_m1), 0)
                    or np.isclose(np.abs(tmp_m1), nrot)
                    or np.isclose(np.abs(tmp_m1), nrot / 2, atol=symprec)
                ):
                    tmp_piH = 0
                    for tmp_piU, tmp_piV in itertools.product(
                        [-1, 1], [-1, 1]
                    ):
                        res = eval_fc(func0, tmp_piU, tmp_piV, tmp_piH)
                        characters.append(_as_complex_array(res))
                        paras_values.append(
                            [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                        )
                else:
                    tmp_piU, tmp_piV = 0, 0
                    for tmp_piH in [-1, 1]:
                        res = eval_fc(func1, tmp_piU, tmp_piV, tmp_piH)
                        characters.append(_as_complex_array(res))
                        paras_values.append(
                            [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                        )
            elif np.isclose(np.abs(tmp_k1), np.pi / a, atol=symprec):
                if np.isclose(np.abs(tmp_m1), nrot / 2, atol=symprec):
                    # Use func2 (k-phase-inclusive 2D rep) instead of
                    # func3 which lacks Bloch phase factors needed for
                    # correct projectors at the BZ boundary.
                    tmp_piU, tmp_piH = 0, 0
                    for tmp_piV in [-1, 1]:
                        res = eval_fc(func2, tmp_piU, tmp_piV, tmp_piH)
                        characters.append(_as_complex_array(res))
                        paras_values.append(
                            [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                        )

                elif np.isclose(np.abs(tmp_m1), 0, atol=symprec):
                    tmp_piU, tmp_piH = 0, 0
                    for tmp_piV in [-1, 1]:
                        res = eval_fc(func2, tmp_piU, tmp_piV, tmp_piH)
                        characters.append(_as_complex_array(res))
                        paras_values.append(
                            [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                        )
                else:
                    tmp_piU, tmp_piV, tmp_piH = 0, 0, 0
                    res = eval_fc(func4, tmp_piU, tmp_piV, tmp_piH)
                    characters.append(_as_complex_array(res))
                    paras_values.append(
                        [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                    )

            elif 0 < np.abs(tmp_k1) < np.pi / a:
                if (
                    np.isclose(np.abs(tmp_m1), 0, atol=symprec)
                    or np.isclose(np.abs(tmp_m1), nrot, atol=symprec)
                    or np.isclose(np.abs(tmp_m1), nrot / 2, atol=symprec)
                ):
                    tmp_piU, tmp_piH = 0, 0
                    for tmp_piV in [-1, 1]:
                        res = eval_fc(func2, tmp_piU, tmp_piV, tmp_piH)
                        characters.append(_as_complex_array(res))
                        paras_values.append(
                            [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                        )

                else:
                    tmp_piU, tmp_piV, tmp_piH = 0, 0, 0
                    res = eval_fc(func4, tmp_piU, tmp_piV, tmp_piH)
                    characters.append(_as_complex_array(res))
                    paras_values.append(
                        [tmp_k1, tmp_m1, tmp_piU, tmp_piV, tmp_piH]
                    )
            else:
                logging.ERROR("Wrong value for k1")
    else:
        raise NotImplementedError("Family %d is not supported yet" % family)
    return characters, paras_values, paras_symbol
