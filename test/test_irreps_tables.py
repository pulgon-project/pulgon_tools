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

from pulgon_tools.character_table_metadata import (
    build_character_table_metadata,
)
from pulgon_tools.generate_irreps_tables import get_linegroup_symmetry_dataset
from pulgon_tools.utils import (
    get_character_num_withparities,
    get_character_withparities,
)

STRUCT_DIR = Path(__file__).parent / "data" / "test_irrep_struct"
EXAMPLE_STRUCT_DIR = Path(__file__).parents[1] / "examples" / "data"

EXPECTED = {
    1: {"atoms": 16, "nrot": 2, "n_axial": 2, "ops": 8, "irreps": 8},
    2: {"atoms": 8, "nrot": 2, "n_axial": 2, "ops": 4, "irreps": 4},
    3: {"atoms": 8, "nrot": 2, "n_axial": 2, "ops": 4, "irreps": 4},
    4: {"atoms": 16, "nrot": 2, "n_axial": 2, "ops": 8, "irreps": 8},
    5: {"atoms": 12, "nrot": 4, "n_axial": 2, "ops": 12, "irreps": 6},
    6: {"atoms": 8, "nrot": 2, "n_axial": 2, "ops": 4, "irreps": 4},
    7: {"atoms": 8, "nrot": 2, "n_axial": 2, "ops": 4, "irreps": 4},
    8: {"atoms": 8, "nrot": 1, "n_axial": 1, "ops": 4, "irreps": 4},
    9: {"atoms": 8, "nrot": 4, "n_axial": 2, "ops": 8, "irreps": 5},
    10: {"atoms": 8, "nrot": 2, "n_axial": 2, "ops": 8, "irreps": 5},
    11: {"atoms": 8, "nrot": 4, "n_axial": 2, "ops": 8, "irreps": 8},
    12: {"atoms": 8, "nrot": 2, "n_axial": 2, "ops": 8, "irreps": 8},
    13: {"atoms": 16, "nrot": 8, "n_axial": 4, "ops": 32, "irreps": 14},
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


def _metadata_from_structure(family: int, qpoint_z: float = 0.0):
    (
        _,
        detected_family,
        nrot,
        a_lattice,
        operations,
        operation_words,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(str(_structure_path(family)))
    params = {
        "qpoints": qpoint_z / a_lattice * 2 * np.pi,
        "nrot": nrot,
        "order": operation_words,
        "family": detected_family,
        "a": a_lattice,
        **gen_angles,
    }
    characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
    metadata = build_character_table_metadata(
        operations,
        operation_words,
        characters,
        a_lattice,
        qpoint_z,
    )
    return characters, metadata


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
        gen_params,
    ) = get_linegroup_symmetry_dataset(atom)

    assert detected_family == family
    assert len(atom_center) == len(atom)
    assert nrot == EXPECTED[family]["nrot"]
    assert a_lattice > 0
    assert len(ops) == EXPECTED[family]["ops"]
    assert len(order_ops) == len(ops)
    assert gen_params["n_axial"] == EXPECTED[family]["n_axial"]


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


@pytest.mark.parametrize("family", [9, 10, 11])
def test_gamma_character_table_satisfies_finite_group_invariants(family):
    params = _params_from_structure(_structure_path(family))
    characters, _, _ = get_character_num_withparities(params, symprec=1e-8)

    dimensions = np.rint(characters[:, 0].real).astype(int)
    assert np.allclose(characters[:, 0].imag, 0.0, atol=1e-10)
    assert np.sum(dimensions**2) == EXPECTED[family]["ops"]

    gram = characters @ characters.conj().T / EXPECTED[family]["ops"]
    np.testing.assert_allclose(gram, np.eye(len(characters)), atol=1e-10)

    rounded_rows = np.round(
        np.column_stack((characters.real, characters.imag)), decimals=10
    )
    assert len(np.unique(rounded_rows, axis=0)) == len(characters)


def test_family_10_bz_boundary_has_two_projective_doublets():
    (
        _,
        family,
        nrot,
        a_lattice,
        operations,
        operation_words,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(str(_structure_path(10)))
    params = {
        "qpoints": np.pi / a_lattice,
        "nrot": nrot,
        "order": operation_words,
        "family": family,
        "a": a_lattice,
        **gen_angles,
    }

    characters, irreps_values, _ = get_character_num_withparities(
        params, symprec=1e-8
    )
    dimensions = np.rint(characters[:, 0].real).astype(int)

    assert len(operations) == 8
    assert characters.shape == (2, 8)
    assert len(irreps_values) == 2
    assert dimensions.tolist() == [2, 2]
    assert np.sum(dimensions**2) == len(operations)
    gram = characters @ characters.conj().T / len(operations)
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)


@pytest.mark.parametrize(
    ("family", "expected_class_sizes"),
    [
        (9, [1, 1, 2, 2, 2]),
        (10, [1, 1, 2, 2, 2]),
        (11, [1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_gamma_conjugacy_class_metadata(family, expected_class_sizes):
    characters, metadata = _metadata_from_structure(family)
    class_ids = metadata["conjugacy_class_ids"]
    class_labels = metadata["conjugacy_class_labels"]
    class_members = metadata["conjugacy_class_members"]
    representatives = metadata["conjugacy_class_representatives"]

    assert metadata["conjugacy_class_scope"] == "finite_factor_group"
    assert metadata["class_characters_available"]
    assert metadata["class_characters_scope"] == (
        "finite_factor_group_at_gamma"
    )
    assert len(metadata["operation_labels"]) == EXPECTED[family]["ops"]
    assert (
        len(np.unique(metadata["operation_labels"])) == EXPECTED[family]["ops"]
    )
    assert sorted(metadata["conjugacy_class_sizes"].tolist()) == (
        expected_class_sizes
    )
    assert sorted(class_members[class_members >= 0].tolist()) == list(
        range(EXPECTED[family]["ops"])
    )
    np.testing.assert_array_equal(
        metadata["operation_class_labels"], class_labels[class_ids]
    )
    np.testing.assert_allclose(
        metadata["class_characters"], characters[:, representatives]
    )

    for class_id, size in enumerate(metadata["conjugacy_class_sizes"]):
        members = class_members[class_id, :size]
        expected = metadata["class_characters"][:, class_id, np.newaxis]
        np.testing.assert_allclose(
            characters[:, members], np.repeat(expected, int(size), axis=1)
        )


def test_non_gamma_metadata_separates_little_group_and_factor_group():
    characters, metadata = _metadata_from_structure(10, qpoint_z=0.25)

    assert metadata["qpoint_z"] == pytest.approx(0.25)
    assert metadata["q_preserving_mask"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    assert metadata["little_group_operation_indices"].tolist() == [0, 1, 5, 7]
    assert metadata["little_group_conjugacy_class_scope"] == (
        "q_preserving_subgroup_of_finite_factor_group"
    )
    assert metadata["little_group_conjugacy_class_sizes"].tolist() == [
        1,
        1,
        1,
        1,
    ]
    assert not metadata["class_characters_available"]
    assert metadata["class_characters_scope"] == "unavailable_non_gamma"
    assert metadata["class_characters"].shape == (characters.shape[0], 0)
    assert np.all(
        metadata["little_group_conjugacy_class_ids"][
            ~metadata["q_preserving_mask"]
        ]
        == -1
    )


@pytest.mark.parametrize(
    ("family", "expected_ops", "expected_dimensions"),
    [
        (10, 12, [1, 1, 1, 1, 2, 2]),
        (11, 12, [1, 1, 1, 1, 2, 2]),
    ],
)
def test_higher_order_examples_satisfy_gamma_invariants(
    family, expected_ops, expected_dimensions
):
    path = EXAMPLE_STRUCT_DIR / f"family_{family:02d}"
    atom = read_vasp(path)
    (
        _,
        detected_family,
        nrot,
        a_lattice,
        ops,
        order_ops,
        gen_params,
    ) = get_linegroup_symmetry_dataset(atom)
    params = {
        "qpoints": 0.0,
        "nrot": nrot,
        "order": order_ops,
        "family": detected_family,
        "a": a_lattice,
        **gen_params,
    }
    characters, _, _ = get_character_num_withparities(params, symprec=1e-8)
    dimensions = np.rint(characters[:, 0].real).astype(int)

    assert detected_family == family
    assert len(ops) == expected_ops
    assert dimensions.tolist() == expected_dimensions
    assert np.sum(dimensions**2) == expected_ops
    gram = characters @ characters.conj().T / expected_ops
    np.testing.assert_allclose(gram, np.eye(len(characters)), atol=1e-10)


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
            "-u",
            "1e-6",
            "-s",
            str(outfile),
        ],
    )

    main()

    data = np.load(f"{outfile}.npz", allow_pickle=False)
    assert data["characters"].shape == (
        EXPECTED[4]["irreps"],
        EXPECTED[4]["ops"],
    )
    assert "ireps_values" in data
    assert "ireps_symbols" in data
    metadata_fields = {
        "qpoint_z",
        "operation_labels",
        "operation_words",
        "conjugacy_class_scope",
        "conjugacy_class_ids",
        "operation_class_labels",
        "conjugacy_class_labels",
        "conjugacy_class_sizes",
        "conjugacy_class_representatives",
        "conjugacy_class_members",
        "class_characters_available",
        "class_characters_scope",
        "class_characters",
        "q_preserving_mask",
        "little_group_operation_indices",
        "little_group_conjugacy_class_scope",
        "little_group_conjugacy_class_ids",
        "little_group_conjugacy_class_labels",
        "little_group_conjugacy_class_sizes",
        "little_group_conjugacy_class_representatives",
        "little_group_conjugacy_class_members",
    }
    assert metadata_fields.issubset(data.files)
    assert data["operation_labels"].shape == (EXPECTED[4]["ops"],)
    assert data["operation_words"].shape == (EXPECTED[4]["ops"],)
    assert data["class_characters"].shape[0] == EXPECTED[4]["irreps"]


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
