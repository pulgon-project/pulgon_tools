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

"""Metadata for interpreting line-group character-table columns."""

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core.operations import SymmOp


def _fractional_affine_operations(
    operations: Sequence[SymmOp], a_lattice: float
) -> np.ndarray:
    """Convert Cartesian ``SymmOp`` translations back to cell fractions."""
    if a_lattice <= 0:
        raise ValueError("a_lattice must be positive")

    affine_operations = []
    for operation in operations:
        affine = np.eye(4)
        affine[:3, :3] = operation.rotation_matrix
        affine[:3, 3] = np.remainder(
            operation.translation_vector / a_lattice, 1.0
        )
        affine_operations.append(affine)
    return np.asarray(affine_operations)


def _compose_affine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the affine operation ``left`` after ``right`` modulo a cell."""
    product = np.eye(4)
    product[:3, :3] = left[:3, :3] @ right[:3, :3]
    product[:3, 3] = np.remainder(
        left[:3, :3] @ right[:3, 3] + left[:3, 3], 1.0
    )
    return product


def _operation_index(
    operation: np.ndarray, operations: np.ndarray, symprec: float
) -> int:
    rotation_error = np.max(
        np.abs(operations[:, :3, :3] - operation[:3, :3]), axis=(1, 2)
    )
    translation_delta = operations[:, :3, 3] - operation[:3, 3]
    translation_delta -= np.rint(translation_delta)
    translation_error = np.max(np.abs(translation_delta), axis=1)
    matches = np.flatnonzero(
        (rotation_error <= symprec) & (translation_error <= symprec)
    )
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely match a product to the finite factor group"
        )
    return int(matches[0])


def _multiplication_table(
    operations: np.ndarray, symprec: float
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build the multiplication table, identity index, and inverse indices."""
    n_operations = len(operations)
    table = np.empty((n_operations, n_operations), dtype=int)
    for left in range(n_operations):
        for right in range(n_operations):
            product = _compose_affine(operations[left], operations[right])
            table[left, right] = _operation_index(product, operations, symprec)

    identity_candidates = []
    expected_indices = np.arange(n_operations)
    for index in range(n_operations):
        if np.array_equal(table[index], expected_indices) and np.array_equal(
            table[:, index], expected_indices
        ):
            identity_candidates.append(index)
    if len(identity_candidates) != 1:
        raise ValueError("Could not uniquely identify the group identity")
    identity = identity_candidates[0]

    inverses = np.empty(n_operations, dtype=int)
    for index in range(n_operations):
        candidates = np.flatnonzero(
            (table[index] == identity) & (table[:, index] == identity)
        )
        if len(candidates) != 1:
            raise ValueError("Could not uniquely identify a group inverse")
        inverses[index] = candidates[0]
    return table, identity, inverses


def _conjugacy_classes(
    table: np.ndarray,
    inverses: np.ndarray,
    member_indices: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    """Partition a group or subgroup into conjugacy classes."""
    if member_indices is None:
        members = np.arange(len(table), dtype=int)
    else:
        members = np.asarray(member_indices, dtype=int)
    member_set = set(members.tolist())

    for left in members:
        for right in members:
            if int(table[left, right]) not in member_set:
                raise ValueError(
                    "The selected little-group operations are not closed"
                )

    classes = []
    unassigned = member_set.copy()
    while unassigned:
        representative = min(unassigned)
        conjugates = {
            int(table[table[h, representative], inverses[h]]) for h in members
        }
        if not conjugates.issubset(member_set):
            raise ValueError("A conjugacy class left the selected subgroup")
        conjugacy_class = np.asarray(sorted(conjugates), dtype=int)
        classes.append(conjugacy_class)
        unassigned.difference_update(conjugates)

    classes.sort(key=lambda conjugacy_class: int(conjugacy_class[0]))
    return classes


def _canonical_vector(vector: np.ndarray, symprec: float) -> np.ndarray:
    vector = np.real_if_close(vector).real
    vector = vector / np.linalg.norm(vector)
    vector[np.abs(vector) <= symprec] = 0.0
    first_nonzero = np.flatnonzero(np.abs(vector) > symprec)[0]
    if vector[first_nonzero] < 0:
        vector *= -1
    return vector


def _eigenvector_for_value(
    rotation: np.ndarray, value: float, symprec: float
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    candidates = np.flatnonzero(np.isclose(eigenvalues, value, atol=symprec))
    if len(candidates) == 0:
        raise ValueError("Could not determine the symmetry-operation axis")
    return _canonical_vector(eigenvectors[:, candidates[0]], symprec)


def _vector_label(vector: np.ndarray, symprec: float) -> str:
    vector = _canonical_vector(vector, symprec)
    basis = np.eye(3)
    for label, direction in zip(("x", "y", "z"), basis):
        if np.allclose(vector, direction, atol=symprec):
            return label

    nonzero = np.abs(vector) > symprec
    scaled = vector / np.min(np.abs(vector[nonzero]))
    integer_scaled = np.rint(scaled).astype(int)
    if (
        np.allclose(scaled, integer_scaled, atol=1e-5)
        and np.max(np.abs(integer_scaled)) <= 9
    ):
        return "[{}]".format(
            ",".join(str(component) for component in integer_scaled)
        )
    return "({:.3f},{:.3f},{:.3f})".format(*vector)


def _signed_angle_degrees(rotation: np.ndarray, axis: np.ndarray) -> float:
    """Return a signed angle for an axial operation when it is well defined."""
    if np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-6):
        return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return float(
        np.degrees(
            np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        )
    )


def _rotation_order(angle_degrees: float) -> int:
    magnitude = abs(angle_degrees)
    if magnitude <= 1e-8:
        return 1
    return max(2, int(round(360.0 / magnitude)))


def _format_angle(angle_degrees: float) -> str:
    rounded = round(angle_degrees)
    if np.isclose(angle_degrees, rounded, atol=1e-6):
        return f"{rounded:+d}deg"
    return f"{angle_degrees:+.3f}deg"


def _format_fraction(value: float, symprec: float) -> str:
    fraction = Fraction(float(value)).limit_denominator(96)
    if np.isclose(float(fraction), value, atol=symprec):
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{value:.6g}"


def _operation_label(operation: np.ndarray, symprec: float) -> str:
    """Create a deterministic, human-readable affine-operation label."""
    rotation = operation[:3, :3]
    determinant = float(np.linalg.det(rotation))

    if np.allclose(rotation, np.eye(3), atol=symprec):
        label = "E"
    elif np.allclose(rotation, -np.eye(3), atol=symprec):
        label = "i"
    elif determinant > 0:
        axis = _eigenvector_for_value(rotation, 1.0, symprec)
        angle = _signed_angle_degrees(rotation, axis)
        order = _rotation_order(angle)
        if order == 2:
            label = f"C2(axis={_vector_label(axis, symprec)})"
        else:
            label = (
                f"C{order}({_format_angle(angle)},"
                f"axis={_vector_label(axis, symprec)})"
            )
    elif np.allclose(
        rotation @ rotation, np.eye(3), atol=symprec
    ) and np.isclose(np.trace(rotation), 1.0, atol=symprec):
        normal = _eigenvector_for_value(rotation, -1.0, symprec)
        if np.allclose(normal, [0.0, 0.0, 1.0], atol=symprec):
            label = "sigma_h"
        else:
            label = f"sigma_v(normal={_vector_label(normal, symprec)})"
    else:
        axis = _eigenvector_for_value(rotation, -1.0, symprec)
        cosine = np.clip((np.trace(rotation) + 1.0) / 2.0, -1.0, 1.0)
        magnitude = float(np.degrees(np.arccos(cosine)))
        if np.allclose(axis, [0.0, 0.0, 1.0], atol=symprec):
            angle = float(
                np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
            )
        else:
            angle = magnitude
        order = _rotation_order(magnitude)
        label = (
            f"S{order}({_format_angle(angle)},"
            f"axis={_vector_label(axis, symprec)})"
        )

    translation = np.remainder(operation[:3, 3], 1.0)
    translation[np.isclose(translation, 1.0, atol=symprec)] = 0.0
    translated_axes = []
    for axis, value in zip(("x", "y", "z"), translation):
        if not np.isclose(value, 0.0, atol=symprec):
            translated_axes.append(
                f"{axis}={_format_fraction(float(value), symprec)}"
            )
    if translated_axes:
        label += " | " + ",".join(translated_axes)
    return label


def _class_arrays(
    classes: Sequence[np.ndarray],
    operation_labels: np.ndarray,
    n_operations: int,
) -> Dict[str, np.ndarray]:
    class_ids = np.full(n_operations, -1, dtype=int)
    class_sizes = np.asarray([len(members) for members in classes], dtype=int)
    representatives = np.asarray(
        [int(members[0]) for members in classes], dtype=int
    )
    max_size = int(max(class_sizes, default=0))
    class_members = np.full((len(classes), max_size), -1, dtype=int)
    class_labels = []
    for class_id, members in enumerate(classes):
        class_ids[members] = class_id
        class_members[class_id, : len(members)] = members
        representative_label = operation_labels[members[0]]
        if len(members) == 1:
            class_labels.append(representative_label)
        else:
            class_labels.append(f"{len(members)} {representative_label}")
    return {
        "ids": class_ids,
        "labels": np.asarray(class_labels, dtype=str),
        "sizes": class_sizes,
        "representatives": representatives,
        "members": class_members,
    }


def build_character_table_metadata(
    operations: Sequence[SymmOp],
    operation_words: Sequence[Sequence[int]],
    characters: np.ndarray,
    a_lattice: float,
    qpoint_z: float,
    symprec: float = 1e-6,
    matrix_tolerance: float = 1e-2,
) -> Dict[str, np.ndarray]:
    """Build operation and conjugacy-class metadata for an NPZ table.

    ``characters`` retains one column per finite-factor-group operation. At a
    Gamma-equivalent q point, this function also verifies that characters are
    constant within every ordinary conjugacy class and returns a compact table.
    At non-Gamma q points, projective phases can invalidate that compression, so
    ``class_characters`` is deliberately empty.
    """
    if len(operations) != len(operation_words):
        raise ValueError(
            "operations and operation_words must have equal length"
        )
    if characters.ndim != 2 or characters.shape[1] != len(operations):
        raise ValueError("characters must have one column per operation")

    affine_operations = _fractional_affine_operations(operations, a_lattice)
    table, _, inverses = _multiplication_table(
        affine_operations, matrix_tolerance
    )
    factor_classes = _conjugacy_classes(table, inverses)
    operation_labels = np.asarray(
        [
            _operation_label(operation, matrix_tolerance)
            for operation in affine_operations
        ],
        dtype=str,
    )
    factor = _class_arrays(factor_classes, operation_labels, len(operations))

    q_vector = np.asarray([0.0, 0.0, qpoint_z])
    q_preserving_mask = np.zeros(len(operations), dtype=bool)
    for index, operation in enumerate(affine_operations):
        q_delta = operation[:3, :3] @ q_vector - q_vector
        q_preserving_mask[index] = np.allclose(
            q_delta, np.rint(q_delta), atol=symprec, rtol=0.0
        )
    little_indices = np.flatnonzero(q_preserving_mask)
    little_classes = _conjugacy_classes(table, inverses, little_indices)
    little = _class_arrays(little_classes, operation_labels, len(operations))

    gamma_equivalent = bool(
        np.isclose(qpoint_z, round(qpoint_z), atol=symprec, rtol=0.0)
    )
    if gamma_equivalent:
        class_characters = characters[:, factor["representatives"]]
        character_tolerance = max(symprec, 1e-10)
        for class_id, members in enumerate(factor_classes):
            expected = class_characters[:, class_id][:, np.newaxis]
            if not np.allclose(
                characters[:, members],
                expected,
                atol=character_tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "Characters are not constant within conjugacy class "
                    f"{factor['labels'][class_id]}"
                )
        class_characters_scope = "finite_factor_group_at_gamma"
    else:
        class_characters = np.empty((characters.shape[0], 0), dtype=complex)
        class_characters_scope = "unavailable_non_gamma"

    operation_class_labels = factor["labels"][factor["ids"]]
    return {
        "qpoint_z": np.asarray(qpoint_z, dtype=float),
        "operation_labels": operation_labels,
        "operation_words": np.asarray(
            [
                ",".join(str(value) for value in word)
                for word in operation_words
            ],
            dtype=str,
        ),
        "conjugacy_class_scope": np.asarray("finite_factor_group"),
        "conjugacy_class_ids": factor["ids"],
        "operation_class_labels": operation_class_labels,
        "conjugacy_class_labels": factor["labels"],
        "conjugacy_class_sizes": factor["sizes"],
        "conjugacy_class_representatives": factor["representatives"],
        "conjugacy_class_members": factor["members"],
        "class_characters_available": np.asarray(gamma_equivalent),
        "class_characters_scope": np.asarray(class_characters_scope),
        "class_characters": class_characters,
        "q_preserving_mask": q_preserving_mask,
        "little_group_operation_indices": little_indices,
        "little_group_conjugacy_class_scope": np.asarray(
            "q_preserving_subgroup_of_finite_factor_group"
        ),
        "little_group_conjugacy_class_ids": little["ids"],
        "little_group_conjugacy_class_labels": little["labels"],
        "little_group_conjugacy_class_sizes": little["sizes"],
        "little_group_conjugacy_class_representatives": little[
            "representatives"
        ],
        "little_group_conjugacy_class_members": little["members"],
    }
