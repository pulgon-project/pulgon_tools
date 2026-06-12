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

from pathlib import Path

import numpy as np
import pytest
from ase.io.vasp import read_vasp

from pulgon_tools.symmetry_projector import (
    get_adapted_matrix_withparities,
    get_linegroup_symmetry_dataset,
)
from pulgon_tools.utils import get_matrices_withPhase

STRUCT_DIR = Path(__file__).parent / "data" / "test_irrep_struct"
FAMILIES = tuple(range(1, 14))

EXPECTED = {
    1: {"atoms": 16, "nrot": 2, "ops": 8},
    2: {"atoms": 8, "nrot": 2, "ops": 4},
    3: {"atoms": 8, "nrot": 2, "ops": 4},
    4: {"atoms": 16, "nrot": 2, "ops": 8},
    5: {"atoms": 12, "nrot": 4, "ops": 12},
    6: {"atoms": 8, "nrot": 2, "ops": 4},
    7: {"atoms": 8, "nrot": 2, "ops": 4},
    8: {"atoms": 8, "nrot": 1, "ops": 4},
    9: {"atoms": 8, "nrot": 4, "ops": 8},
    10: {"atoms": 8, "nrot": 2, "ops": 4},
    11: {"atoms": 8, "nrot": 4, "ops": 8},
    12: {"atoms": 8, "nrot": 2, "ops": 8},
    13: {"atoms": 16, "nrot": 8, "ops": 32},
}

KNOWN_ADAPTED_ERRORS = {}

# Family 5 needs a slightly relaxed permutation tolerance: the detected
# f_screw = 3.001 carries a ~1e-3 rounding error inherited from the
# 4-decimal z fractional coordinates in family_05.vasp, and screw^2
# accumulates that to ~2e-3.  Other families use the default 1e-3.
FAMILY_SYMPREC = {5: 2e-3}

KNOWN_INCOMPLETE_OR_NONORTHONORMAL = {}


def _structure_path(family: int) -> Path:
    return STRUCT_DIR / f"family_{family:02d}.vasp"


def _qpoint(label: str, a_lattice: float) -> float:
    if label == "gamma":
        return 0.0
    if label == "nonzero_q":
        return 0.25 * 2 * np.pi / a_lattice
    if label == "bz_boundary":
        return 0.5 * 2 * np.pi / a_lattice
    raise ValueError(f"Unknown q-point label: {label}")


def _dataset_for_family(family: int) -> dict:
    path = _structure_path(family)
    atom = read_vasp(path)
    (
        atom_center,
        detected_family,
        nrot,
        a_lattice,
        ops_car_sym,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(str(path))
    return {
        "path": path,
        "atom": atom,
        "atom_center": atom_center,
        "family": detected_family,
        "nrot": nrot,
        "a_lattice": a_lattice,
        "ops_car_sym": ops_car_sym,
        "order_ops": order_ops,
        "gen_angles": gen_angles,
        "num_atom": len(atom_center),
        "ndof": 3 * len(atom_center),
    }


def _adapted_result(family: int, q_label: str) -> dict:
    ds = _dataset_for_family(family)
    qpoint = _qpoint(q_label, ds["a_lattice"])
    dict_params = {
        "qpoints": qpoint,
        "nrot": ds["nrot"],
        "order": ds["order_ops"],
        "family": ds["family"],
        "a": ds["a_lattice"],
        **ds["gen_angles"],
    }
    symprec = FAMILY_SYMPREC.get(family, 1e-3)
    try:
        matrices = get_matrices_withPhase(
            ds["atom_center"], ds["ops_car_sym"], qpoint, symprec=symprec
        )
        (
            adapted,
            dimensions,
            paras_values,
            paras_symbols,
        ) = get_adapted_matrix_withparities(
            dict_params, ds["num_atom"], matrices
        )
    except Exception as exc:
        reason = KNOWN_ADAPTED_ERRORS.get((family, q_label))
        if reason:
            pytest.xfail(f"{reason}: {type(exc).__name__}: {exc}")
        raise

    return {
        **ds,
        "q_label": q_label,
        "qpoint": qpoint,
        "dict_params": dict_params,
        "matrices": matrices,
        "adapted": adapted,
        "dimensions": dimensions,
        "paras_values": paras_values,
        "paras_symbols": paras_symbols,
    }


@pytest.fixture(params=FAMILIES, ids=lambda family: f"family_{family:02d}")
def family(request):
    return request.param


@pytest.fixture
def adapted_result_gamma(family):
    return _adapted_result(family, "gamma")


@pytest.fixture
def adapted_result_nonzero_q(family):
    return _adapted_result(family, "nonzero_q")


@pytest.fixture
def adapted_result_bz_boundary(family):
    return _adapted_result(family, "bz_boundary")


@pytest.mark.parametrize(
    "family", FAMILIES, ids=lambda family: f"family_{family:02d}"
)
def test_linegroup_dataset_from_test_irrep_struct(family):
    ds = _dataset_for_family(family)

    assert ds["path"].is_file()
    assert len(ds["atom"]) == EXPECTED[family]["atoms"]
    assert ds["family"] == family
    assert ds["nrot"] == EXPECTED[family]["nrot"]
    assert len(ds["ops_car_sym"]) == EXPECTED[family]["ops"]
    assert len(ds["order_ops"]) == EXPECTED[family]["ops"]
    assert ds["a_lattice"] > 0


def test_linegroup_dataset_accepts_tolerance_parameters():
    path = _structure_path(1)
    (
        atom_center,
        detected_family,
        nrot,
        a_lattice,
        ops_car_sym,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(
        str(path),
        tolerance=1e-2,
        layer_tolerance=0.05,
        matrix_tolerance=1e-2,
    )

    assert len(atom_center) == EXPECTED[1]["atoms"]
    assert detected_family == 1
    assert nrot == EXPECTED[1]["nrot"]
    assert a_lattice > 0
    assert len(ops_car_sym) == EXPECTED[1]["ops"]
    assert len(order_ops) == EXPECTED[1]["ops"]
    assert gen_angles


def test_adapted_matrix_accepts_projector_tolerances():
    ds = _dataset_for_family(1)
    qpoint = 0.0
    dict_params = {
        "qpoints": qpoint,
        "nrot": ds["nrot"],
        "order": ds["order_ops"],
        "family": ds["family"],
        "a": ds["a_lattice"],
        **ds["gen_angles"],
    }
    matrices = get_matrices_withPhase(
        ds["atom_center"], ds["ops_car_sym"], qpoint, symprec=1e-3
    )

    adapted, dimensions, _, _ = get_adapted_matrix_withparities(
        dict_params,
        ds["num_atom"],
        matrices,
        rank_tolerance=1e-8,
        gap_warning_tolerance=0.05,
    )

    assert adapted.shape == (ds["ndof"], ds["ndof"])
    assert sum(dimensions) == ds["ndof"]


def _assert_adapted_result_smoke(result):
    adapted = result["adapted"]
    dimensions = result["dimensions"]

    assert adapted.shape[0] == result["ndof"]
    assert adapted.shape[1] == sum(dimensions)
    assert np.isfinite(adapted.real).all()
    assert np.isfinite(adapted.imag).all()
    assert len(dimensions) > 0
    assert all(isinstance(dim, (int, np.integer)) for dim in dimensions)
    assert all(dim > 0 for dim in dimensions)
    assert len(result["paras_values"]) >= len(dimensions)
    assert len(result["paras_symbols"]) > 0
    assert len(result["matrices"]) == len(result["ops_car_sym"])
    assert all(
        matrix.shape == (result["ndof"], result["ndof"])
        for matrix in result["matrices"]
    )


def test_adapted_result_gamma_smoke(adapted_result_gamma):
    _assert_adapted_result_smoke(adapted_result_gamma)


def test_adapted_result_nonzero_q_smoke(adapted_result_nonzero_q):
    _assert_adapted_result_smoke(adapted_result_nonzero_q)


def test_adapted_result_bz_boundary_smoke(adapted_result_bz_boundary):
    _assert_adapted_result_smoke(adapted_result_bz_boundary)


def _assert_complete_orthonormal_basis(result):
    issue = KNOWN_INCOMPLETE_OR_NONORTHONORMAL.get(
        (result["family"], result["q_label"])
    )
    if issue:
        pytest.xfail(issue)

    adapted = result["adapted"]
    dimensions = result["dimensions"]
    ndof = result["ndof"]

    assert adapted.shape == (ndof, ndof)
    assert sum(dimensions) == ndof
    np.testing.assert_allclose(
        adapted.conj().T @ adapted,
        np.eye(ndof),
        atol=1e-8,
    )


def test_adapted_result_gamma_is_complete_orthonormal_basis(
    adapted_result_gamma,
):
    _assert_complete_orthonormal_basis(adapted_result_gamma)


def test_adapted_result_nonzero_q_is_complete_orthonormal_basis(
    adapted_result_nonzero_q,
):
    _assert_complete_orthonormal_basis(adapted_result_nonzero_q)


def test_adapted_result_bz_boundary_is_complete_orthonormal_basis(
    adapted_result_bz_boundary,
):
    _assert_complete_orthonormal_basis(adapted_result_bz_boundary)
