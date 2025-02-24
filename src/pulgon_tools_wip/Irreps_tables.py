# Copyright 2023 The PULGON Project Developers
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
import itertools
import logging
import math
import typing
from pdb import set_trace

import numpy as np
import sympy
from sympy import symbols
from sympy.ntheory.factor_ import totient
from tqdm import tqdm


def frac_range(
    start: float,
    end: float,
    left: bool = True,
    right: bool = True,
    symprec: float = 0.01,
) -> list:
    """return the integer within the specified range

    Args:
        start: left boundary
        end: right boundary
        left: False mean delete the left boundary element if it is an integer
        right: False mean delete the right boundary element if it is an integer
        symprec: system precise

    Returns:

    """
    close = list(
        range(
            np.ceil(start).astype(np.int32), np.floor(end).astype(np.int32) + 1
        )
    )
    if left == False:
        if close[0] - start < symprec:
            close.pop(0)  # delete the left boundary
    if right == False:
        if close[-1] - end < symprec:
            close.pop()  # delete the right boundary
    return close


def get_modified_Dmu(DictParams, tmp_m1, symprec=1e-6):
    family = DictParams["family"]

    if family == 4:
        nrot = DictParams["nrot"]
        tmp_k1 = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        aL = DictParams["a"]

        k1, m1, n, a, piH, t, s, j = symbols("k1 m1 n a piH t s j")
        func0 = sympy.Matrix(
            [
                sympy.exp(1j * m1 * sympy.pi / n) ** t,
                sympy.exp(1j * m1 * 2 * sympy.pi / n) ** s,
                piH**j,
            ]
        )

        func1 = [
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
            )
            ** t,
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
            )
            ** s,
            sympy.Matrix([[0, 1], [1, 0]]) ** j,
        ]

        func = [func0, func1]
        if np.isclose(tmp_k1, 0, atol=symprec):
            fc = func[0]
            for ii, tmp in enumerate(range(1, 3)):
                if ii == 0:
                    Dmu_rot_symbol = fc[tmp]
                else:
                    Dmu_rot_symbol = Dmu_rot_symbol * fc[tmp]
            Dmu_rot = []
            for s in range(nrot):
                for j in range(2):
                    tmp = Dmu_rot_symbol.subs(
                        {
                            "k1": tmp_k1,
                            "m1": tmp_m1,
                            "n": nrot,
                            "a": aL,
                            "t": 0,
                            "s": s,
                            "j": j,
                        }
                    ).conjugate()

                    Dmu_rot.append(
                        np.array(
                            (tmp.subs({"piH": 1}) + tmp.subs({"piH": -1}))
                        ).astype(np.complex128)
                    )
        else:
            fc = func[1]

            for ii, tmp in enumerate(range(1, 3)):
                if ii == 0:
                    Dmu_rot_symbol = fc[tmp]
                else:
                    Dmu_rot_symbol = Dmu_rot_symbol * fc[tmp]
            Dmu_rot = []

            for s in range(nrot):
                for j in range(2):
                    Dmu_rot.append(
                        np.array(
                            Dmu_rot_symbol.subs(
                                {
                                    "k1": tmp_k1,
                                    "m1": tmp_m1,
                                    "n": nrot,
                                    "a": aL,
                                    "t": 0,
                                    "s": s,
                                    "j": j,
                                }
                            ).conjugate()
                        ).astype(np.complex128)
                    )

        Dmu_tran_symbol = fc[0]
        Dmu_tran = np.array(
            Dmu_tran_symbol.subs(
                {"k1": tmp_k1, "m1": tmp_m1, "n": nrot, "a": aL, "t": 1}
            ).conjugate()
        ).astype(np.complex128)
    elif family == 2:
        qpoints = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        aL = DictParams["a"]
        tmp_k1 = DictParams["qpoints"]

        k1, m1, n, piH, a, t, s = symbols("k1 m1 n piH a t s")
        func0 = sympy.Matrix(
            [
                sympy.exp(1j * k1 * a) ** t,
                (piH * sympy.exp(1j * m1 * sympy.pi / n)) ** s,
            ]
        )
        func1 = [
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
            )
            ** t,
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
            )
            ** s,
        ]

        func = [func0, func1]
        if np.isclose(tmp_k1, 0, atol=symprec) or np.isclose(
            tmp_k1, np.pi / aL, atol=symprec
        ):
            fc = func[0]
            Dmu_rot_symbol = fc[1]
            Dmu_rot = []
            for s in range(2 * nrot):
                tmp = Dmu_rot_symbol.subs(
                    {
                        "k1": tmp_k1,
                        "m1": tmp_m1,
                        "n": nrot,
                        "a": aL,
                        "t": 0,
                        "s": s,
                    }
                ).conjugate()
                Dmu_rot.append(
                    np.array(
                        (tmp.subs({"piH": 1}) + tmp.subs({"piH": -1}))
                    ).astype(np.complex128)
                )
        else:
            fc = func[1]
            Dmu_rot_symbol = fc[1]
            Dmu_rot = []
            for s in range(2 * nrot):
                Dmu_rot.append(
                    np.array(
                        Dmu_rot_symbol.subs(
                            {
                                "k1": tmp_k1,
                                "m1": tmp_m1,
                                "n": nrot,
                                "a": aL,
                                "t": 0,
                                "s": s,
                            }
                        ).conjugate()
                    ).astype(np.complex128)
                )

        Dmu_tran_symbol = fc[0]
        Dmu_tran = np.array(
            Dmu_tran_symbol.subs(
                {
                    "k1": tmp_k1,
                    "m1": tmp_m1,
                    "n": nrot,
                    "a": aL,
                    "t": 1,
                    "s": 0,
                }
            ).conjugate()
        ).astype(np.complex128)
    return Dmu_rot, Dmu_tran


def line_group_sympy(DictParams, symprec=1e-6):
    family = DictParams["family"]
    if family == 2:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]

        k1, m1, n, piH = symbols("k1 m1 n piH")
        func0 = sympy.Matrix(
            [
                1,
                # sympy.exp(1j * k1 * a),
                piH * sympy.exp(1j * m1 * sympy.pi / n),
            ]
        )
        func1 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            # sympy.Matrix(
            #     [
            #         [
            #             sympy.exp(1j * k1 * a),
            #             0,
            #         ],
            #         [
            #             0,
            #             sympy.exp(-1j * k1 * a),
            #         ],
            #     ]
            # ),
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

        func = [func0, func1]
        qps_value = [qpoint]
        n_value = [nrot]
        m1_value = list(range(int(-nrot / 2) + 1, int(nrot / 2) + 1))
        piH_value = [-1, 1]

        if np.isclose(qpoint, 0, atol=symprec) or np.isclose(
            qpoint, np.pi / a, atol=symprec
        ):
            fc = func[0]
            paras_symbol = [k1, m1, n, piH]
        else:
            fc = func[1]
            paras_symbol = [k1, m1, n]
        paras_values = list(itertools.product(*[qps_value, m1_value, n_value]))
        characters = []
        for ii, paras_value in enumerate(paras_values):
            tmp_k1, tmp_m1, tmp_n = paras_value
            res = []
            for tmp_order in order:
                for jj, tmp in enumerate(tmp_order):
                    if jj == 0:
                        tmp0 = fc[tmp]
                    else:
                        tmp0 = tmp0 * fc[tmp]
                        # tmp0 = fc[tmp] * tmp0
                        set_trace()
                if len(paras_symbol) == 4:
                    tmp1 = tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piH: -1}
                    ) + tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piH: 1}
                    )
                    res.append(tmp1)
                else:
                    tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n})
                    res.append(tmp1)
            characters.append(np.array(res).astype(np.complex128))
        # characters = np.array(characters).astype(np.complex128)
    elif family == 3:
        qpoint = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]

        k1, m1, n, piH = symbols("k1 m1 n piH")
        func0 = sympy.Matrix(
            [
                1,
                # sympy.exp(1j * k1 * a),
                sympy.exp(1j * m1 * 2 * sympy.pi / n),
                piH,
            ]
        )
        func1 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            # sympy.Matrix(
            #     [
            #         [
            #             sympy.exp(1j * k1 * a),
            #             0,
            #         ],
            #         [
            #             0,
            #             sympy.exp(-1j * k1 * a),
            #         ],
            #     ]
            # ),
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

        func = [func0, func1]
        qps_value = [qpoint]
        n_value = [nrot]
        m1_value = list(range(int(-nrot / 2) + 1, int(nrot / 2) + 1))
        # piH_value = [-1, 1]

        if np.isclose(qpoint, 0, atol=symprec) or np.isclose(
            qpoint, np.pi / a, atol=symprec
        ):
            fc = func[0]
            paras_symbol = [k1, m1, n, piH]
        else:
            fc = func[1]
            paras_symbol = [k1, m1, n]
        paras_values = list(itertools.product(*[qps_value, m1_value, n_value]))
        characters = []
        for ii, paras_value in enumerate(paras_values):
            tmp_k1, tmp_m1, tmp_n = paras_value
            res = []
            for tmp_order in order:
                for jj, tmp in enumerate(tmp_order):
                    if jj == 0:
                        tmp0 = fc[tmp]
                    else:
                        tmp0 = tmp0 * fc[tmp]
                if len(paras_symbol) == 4:
                    tmp1 = tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piH: -1}
                    ) + tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piH: 1}
                    )
                    res.append(tmp1)
                else:
                    tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n})
                    res.append(tmp1)
            characters.append(np.array(res).astype(np.complex128))
        # characters = np.array(characters).astype(np.complex128)
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
                        # sympy.exp(1j * (m1 * sympy.pi / n)),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * (m1 * sympy.pi / n - k1 * a / 2)),
                        # sympy.exp(1j * (m1 * sympy.pi / n)),
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

        func = [func0, func1]
        qps_value = [qpoint]
        n_value = [nrot]

        m1_value = list(range(-nrot + 1, nrot + 1))

        piH_value = [-1, 1]
        if np.isclose(qpoint, 0, atol=symprec):
            fc = func[0]
            paras_symbol = [k1, m1, n, piH]
        else:
            fc = func[1]
            paras_symbol = [k1, m1, n]
        paras_values = list(itertools.product(*[qps_value, m1_value, n_value]))
        characters = []
        for ii, paras_value in enumerate(paras_values):
            tmp_k1, tmp_m1, tmp_n = paras_value
            res = []
            for tmp_order in order:
                for jj, tmp in enumerate(tmp_order):
                    if jj == 0:
                        tmp0 = fc[tmp]
                    else:
                        tmp0 = tmp0 * fc[tmp]
                if len(paras_symbol) == 4:
                    tmp1 = tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piH: -1}
                    ) + tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piH: 1}
                    )
                    res.append(tmp1)
                else:
                    tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n})
                    res.append(tmp1)
            characters.append(np.array(res).astype(np.complex128))
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

        func = [func0, func1]
        qps_value = [qpoint]
        n_value = [nrot]
        m1_value = list(range(0, int(nrot / 2) + 1))
        # piV_value = [-1, 1]

        paras_values = list(itertools.product(*[qps_value, m1_value, n_value]))
        characters = []
        for ii, paras_value in enumerate(paras_values):
            tmp_k1, tmp_m1, tmp_n = paras_value

            if np.isclose(tmp_m1, 0, atol=symprec) or np.isclose(
                tmp_m1, tmp_n / 2, atol=symprec
            ):
                fc = func[0]
                paras_symbol = [k1, m1, n, piV]
            else:
                fc = func[1]
                paras_symbol = [k1, m1, n]

            res = []
            for tmp_order in order:
                for jj, tmp in enumerate(tmp_order):
                    if jj == 0:
                        tmp0 = fc[tmp]
                    else:
                        # tmp0 = tmp0 * fc[tmp]
                        tmp0 = fc[tmp] * tmp0
                if len(paras_symbol) == 4:
                    tmp1 = tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: -1}
                    ) + tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: 1}
                    )
                    res.append(tmp1)
                else:
                    tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n})
                    res.append(tmp1)
            characters.append(np.array(res).astype(np.complex128))
        # characters = np.array(characters).astype(np.complex128)
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
                # sympy.exp(1j * (m1 * sympy.pi / n)),
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
                        # sympy.exp(1j * (m1 * sympy.pi / n)),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(1j * (k1 * a / 2 - m1 * sympy.pi / n)),
                        # sympy.exp(1j * (-m1 * sympy.pi / n)),
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

        func = [func0, func1]
        qps_value = [qpoint]
        n_value = [nrot]

        m1_value = list(range(0, nrot + 1))
        paras_values = list(itertools.product(*[qps_value, m1_value, n_value]))
        characters = []
        for ii, paras_value in enumerate(paras_values):
            tmp_k1, tmp_m1, tmp_n = paras_value

            if np.isclose(tmp_m1, 0, atol=symprec) or np.isclose(
                tmp_m1, nrot, atol=symprec
            ):
                fc = func[0]
                paras_symbol = [k1, m1, n, piV]
            else:
                fc = func[1]
                paras_symbol = [k1, m1, n]

            res = []
            for tmp_order in order:
                for jj, tmp in enumerate(tmp_order):
                    if jj == 0:
                        tmp0 = fc[tmp]
                    else:
                        # tmp0 = tmp0 * fc[tmp]
                        tmp0 = fc[tmp] * tmp0
                if len(paras_symbol) == 4:
                    tmp1 = tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: -1}
                    ) + tmp0.evalf(
                        subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: 1}
                    )
                    # tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: 0})
                    res.append(tmp1)
                else:
                    tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n})
                    res.append(tmp1)
            characters.append(np.array(res).astype(np.complex128))
    elif family == 13:
        qpoints = DictParams["qpoints"]
        nrot = DictParams["nrot"]
        a = DictParams["a"]
        order = DictParams["order"]

        n, k1, m1, f, piU, piV, piH = symbols("n k1 m1 f piU piV piH")
        func0 = sympy.Matrix(
            [
                1,
                sympy.exp(1j * m1 * sympy.pi / n),
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
                    [sympy.exp(1j * m1 * sympy.pi / n), 0],
                    [0, sympy.exp(-1j * m1 * sympy.pi / n)],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        sympy.exp(1j * 2 * m1 * sympy.pi / n),
                        0,
                    ],
                    [
                        0,
                        sympy.exp(-1j * 2 * m1 * sympy.pi / n),
                    ],
                ]
            ),
            sympy.Matrix(
                [
                    [
                        0,
                        piH,
                    ],
                    [
                        piH,
                        0,
                    ],
                ]
            ),
            sympy.Matrix([[0, 1], [1, 0]]),
        ]
        func2 = [
            sympy.Matrix([[1, 0], [0, 1]]),
            sympy.Matrix(
                [
                    [sympy.exp(1j * (m1 * sympy.pi / n + k1 * f)), 0],
                    [0, sympy.exp(1j * (m1 * sympy.pi / n - k1 * f))],
                ]
            ),
            sympy.Matrix([[0, 1], [1, 0]]),
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
                    [sympy.exp(1j * (m1 * sympy.pi / n + k1 * f)), 0, 0, 0],
                    [0, sympy.exp(1j * (-m1 * sympy.pi / n + k1 * f)), 0, 0],
                    [0, 0, sympy.exp(1j * (m1 * sympy.pi / n - k1 * f)), 0],
                    [0, 0, 0, sympy.exp(-1j * (m1 * sympy.pi / n + k1 * f))],
                ]
            ),
            sympy.Matrix(
                [
                    [sympy.exp(1j * 2 * sympy.pi * m1 / n), 0, 0, 0],
                    [0, sympy.exp(-1j * 2 * sympy.pi * m1 / n), 0, 0],
                    [0, 0, sympy.exp(1j * 2 * sympy.pi * m1 / n), 0],
                    [0, 0, 0, sympy.exp(-1j * 2 * sympy.pi * m1 / n)],
                ]
            ),
            sympy.Matrix(
                [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
            ),
            sympy.Matrix(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
            ),
        ]
        func = [func0, func1, func2, func3, func4]

        qps_value = [qpoints]
        n_value = [nrot]
        m1_value = list(range(0, nrot + 1))
        # m1_value = list(range(-nrot + 1, nrot + 1))
        f_value = [a / 2]

        paras_values = list(
            itertools.product(
                *[
                    qps_value,
                    m1_value,
                    n_value,
                    f_value,
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                ]
            )
        )
        paras_symbol = [k1, m1, n, f, piU, piV, piH]

        print("Now getting characters:")
        characters = []
        for ii, paras_value in enumerate(paras_values):
            tmp_k1, tmp_m1, tmp_n, tmp_f, tmppiU, tmppiV, tmppiH = paras_value

            if np.isclose(tmp_k1, 0, atol=symprec):
                if np.isclose(np.abs(tmp_m1), 0) or np.isclose(
                    np.abs(tmp_m1), tmp_n
                ):
                    fc = func[0]
                elif 0 < np.abs(tmp_m1) < tmp_n:
                    fc = func[1]
                else:
                    logging.ERROR("Wrong value for m1")
            elif np.isclose(np.abs(tmp_k1), np.pi / a, atol=symprec):
                if np.isclose(np.abs(tmp_m1), tmp_n / 2, atol=symprec):
                    fc = func[3]
                elif np.isclose(np.abs(tmp_m1), 0, atol=symprec):
                    fc = func[2]
                elif 0 < np.abs(tmp_m1) < tmp_n / 2:
                    fc = func[4]
                else:
                    logging.ERROR("Wrong value for m1")
            elif 0 < np.abs(tmp_k1) < np.pi / a:
                if np.isclose(np.abs(tmp_m1), 0, atol=symprec) or np.isclose(
                    np.abs(tmp_m1), tmp_n, atol=symprec
                ):
                    fc = func[2]
                elif 0 < np.abs(tmp_m1) < tmp_n:
                    fc = func[4]
                else:
                    logging.ERROR("Wrong value for m1")
            else:
                # set_trace()
                logging.ERROR("Wrong value for k1")

            res = []
            for tmp_order in order:
                for jj, tmp in enumerate(tmp_order):
                    if jj == 0:
                        tmp0 = fc[tmp]
                    else:
                        tmp0 = tmp0 * fc[tmp]

                tmp1 = tmp0.evalf(
                    subs={
                        k1: tmp_k1,
                        m1: tmp_m1,
                        n: tmp_n,
                        f: tmp_f,
                        piU: tmppiU,
                        piV: tmppiV,
                        piH: tmppiH,
                    }
                )
                if tmp1.is_Matrix:
                    res.append(tmp1.trace())
                else:
                    res.append(tmp1)
                # res.append(tmp1)
            res = np.array(res).astype(np.complex128)
            characters.append(res)
    else:
        raise NotImplementedError("Family %d is not supported yet" % family)
    return characters, paras_values, paras_symbol


# def get_IR_matrix(DictParams, symprec=1e-6):
#     family = DictParams["family"]
#
#     if family == 8:
#         qpoint = DictParams["qpoints"]
#         nrot = DictParams["nrot"]
#         a = DictParams["a"]
#         order = DictParams["order"]
#
#         k1, m1, n, piV = symbols("k1 m1 n piV")
#         func0 = sympy.Matrix(
#             [
#                 1,
#                 # sympy.exp(1j * (k1 * a / 2 + m1 * sympy.pi / n)),
#                 sympy.exp(1j * (m1 * sympy.pi / n)),
#                 1,
#                 piV,
#             ]
#         )
#
#         func1 = [
#             sympy.Matrix([[1, 0], [0, 1]]),
#             sympy.Matrix(
#                 [
#                     [
#                         # sympy.exp(1j * (k1 * a / 2 + m1 * sympy.pi / n)),
#                         sympy.exp(1j * (m1 * sympy.pi / n)),
#                         0,
#                     ],
#                     [
#                         0,
#                         # sympy.exp(1j * (k1 * a / 2 - m1 * sympy.pi / n)),
#                         sympy.exp(1j * (-m1 * sympy.pi / n)),
#                     ],
#                 ]
#             ),
#             sympy.Matrix(
#                 [
#                     [
#                         sympy.exp(1j * m1 * 2 * sympy.pi / n),
#                         0,
#                     ],
#                     [
#                         0,
#                         sympy.exp(-1j * m1 * 2 * sympy.pi / n),
#                     ],
#                 ]
#             ),
#             sympy.Matrix([[0, 1], [1, 0]]),
#         ]
#
#         func = [func0, func1]
#         qps_value = [qpoint]
#         n_value = [nrot]
#
#         m1_value = list(range(0, nrot + 1))
#         paras_values = list(itertools.product(*[qps_value, m1_value, n_value]))
#         IR_matrix = []
#         for ii, paras_value in enumerate(paras_values):
#             tmp_k1, tmp_m1, tmp_n = paras_value
#
#             if np.isclose(tmp_m1, 0, atol=symprec) or np.isclose(
#                 tmp_m1, nrot, atol=symprec
#             ):
#                 fc = func[0]
#                 paras_symbol = [k1, m1, n, piV]
#             else:
#                 fc = func[1]
#                 paras_symbol = [k1, m1, n]
#
#             res = []
#             for tmp_order in order:
#                 for jj, tmp in enumerate(tmp_order):
#                     if jj == 0:
#                         tmp0 = fc[tmp]
#                     else:
#                         # tmp0 = tmp0 * fc[tmp]
#                         tmp0 = fc[tmp] * tmp0
#                 if len(paras_symbol) == 4:
#                     tmp1 = tmp0.evalf(
#                         subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: -1}
#                     ) + tmp0.evalf(
#                         subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: 1}
#                     )
#                     # tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n, piV: 1})
#                     res.append(tmp1)
#                 else:
#                     tmp1 = tmp0.evalf(subs={k1: tmp_k1, m1: tmp_m1, n: tmp_n})
#                     res.append(tmp1)
#             IR_matrix.append(np.array(res).astype(np.complex128))
#     return IR_matrix, paras_values, paras_symbol
