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

from pulgon_tools.generate_irreps_tables import get_linegroup_symmetry_dataset
from pulgon_tools.utils import (
    get_character_num_withparities,
    get_character_withparities,
)

STRUCT_DIR = Path(__file__).parent / "data" / "test_irrep_struct"

EXPECTED = {
    1: {"atoms": 12, "nrot": 2, "ops": 6, "irreps": 6},
    2: {"atoms": 8, "nrot": 2, "ops": 2, "irreps": 4},
    3: {"atoms": 8, "nrot": 2, "ops": 4, "irreps": 4},
    4: {"atoms": 12, "nrot": 2, "ops": 4, "irreps": 8},
    5: {"atoms": 12, "nrot": 4, "ops": 12, "irreps": 8},
    6: {"atoms": 8, "nrot": 2, "ops": 4, "irreps": 4},
    7: {"atoms": 8, "nrot": 2, "ops": 4, "irreps": 4},
    8: {"atoms": 8, "nrot": 1, "ops": 4, "irreps": 4},
    9: {"atoms": 8, "nrot": 4, "ops": 8, "irreps": 18},
    10: {"atoms": 8, "nrot": 2, "ops": 4, "irreps": 5},
    11: {"atoms": 8, "nrot": 4, "ops": 8, "irreps": 18},
    12: {"atoms": 8, "nrot": 2, "ops": 8, "irreps": 8},
    13: {"atoms": 16, "nrot": 2, "ops": 8, "irreps": 8},
}


def _structure_path(family: int) -> Path:
    return STRUCT_DIR / f"family_{family:02d}.vasp"


def _params_from_structure(path: Path) -> dict:
    atom = read_vasp(path)
    (
        _,
        family,
        nrot,
        a_lattice,
        _,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(atom)
    return {
        "qpoints": 0.0,
        "nrot": nrot,
        "order": order_ops,
        "family": family,
        "a": a_lattice,
        **gen_angles,
    }


@pytest.mark.parametrize("family", EXPECTED)
def test_test_irrep_struct_files_exist(family):
    path = _structure_path(family)

    assert path.is_file()
    assert len(read_vasp(path)) == EXPECTED[family]["atoms"]


@pytest.mark.parametrize("family", EXPECTED)
def test_linegroup_symmetry_dataset_uses_test_irrep_struct(family):
    atom = read_vasp(_structure_path(family))
    (
        atom_center,
        detected_family,
        nrot,
        a_lattice,
        ops,
        order_ops,
        _,
    ) = get_linegroup_symmetry_dataset(atom)

    assert detected_family == family
    assert len(atom_center) == len(atom)
    assert nrot == EXPECTED[family]["nrot"]
    assert a_lattice > 0
    assert len(ops) == EXPECTED[family]["ops"]
    assert len(order_ops) == len(ops)


@pytest.mark.parametrize("family", EXPECTED)
def test_character_table_shape_for_test_irrep_struct(family):
    params = _params_from_structure(_structure_path(family))

    characters, irreps_values, irreps_symbols = get_character_num_withparities(
        params, symprec=1e-8
    )

    assert characters.shape == (
        EXPECTED[family]["irreps"],
        EXPECTED[family]["ops"],
    )
    assert len(irreps_values) == characters.shape[0]
    assert len(irreps_symbols) > 0
    assert np.isfinite(characters.real).all()
    assert np.isfinite(characters.imag).all()


@pytest.mark.parametrize("family", EXPECTED)
def test_representation_traces_match_characters(family):
    params = _params_from_structure(_structure_path(family))

    representation_matrices, _, _ = get_character_withparities(
        params, symprec=1e-8
    )
    characters, _, _ = get_character_num_withparities(params, symprec=1e-8)

    assert len(representation_matrices) == characters.shape[0]
    for idx, rep in enumerate(representation_matrices):
        if rep.ndim == 1:
            assert rep.shape == (characters.shape[1],)
            trace = rep
        else:
            assert rep.shape[0] == characters.shape[1]
            assert rep.shape[1] == rep.shape[2]
            trace = np.trace(rep, axis1=1, axis2=2)
        assert np.allclose(trace, characters[idx], atol=1e-10)


def test_dataset_accepts_path_string():
    _, family, nrot, _, ops, _, _ = get_linegroup_symmetry_dataset(
        str(_structure_path(8))
    )

    assert family == 8
    assert nrot == EXPECTED[8]["nrot"]
    assert len(ops) == EXPECTED[8]["ops"]


def test_main_cli_saves_character_table(tmp_path, monkeypatch):
    import sys

    from pulgon_tools.generate_irreps_tables import main

    outfile = tmp_path / "family_04_chars"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pulgon-irreps-tables",
            "-p",
            str(_structure_path(4)),
            "-q",
            "0.0",
            "-s",
            str(outfile),
        ],
    )

    main()

    data = np.load(f"{outfile}.npz", allow_pickle=True)
    assert data["characters"].shape == (
        EXPECTED[4]["irreps"],
        EXPECTED[4]["ops"],
    )
    assert "ireps_values" in data
    assert "ireps_symbols" in data


def test_main_cli_saves_representation_matrices(tmp_path, monkeypatch):
    import sys

    from pulgon_tools.generate_irreps_tables import main

    outfile = tmp_path / "family_13_reps"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pulgon-irreps-tables",
            "-p",
            str(_structure_path(13)),
            "-q",
            "0.0",
            "-s",
            str(outfile),
            "-r",
        ],
    )

    main()

    data = np.load(f"{outfile}.npz", allow_pickle=True)
    rep_keys = [key for key in data.files if key.startswith("D_irrep_")]
    assert data["characters"].shape == (
        EXPECTED[13]["irreps"],
        EXPECTED[13]["ops"],
    )
    assert len(rep_keys) == EXPECTED[13]["irreps"]
