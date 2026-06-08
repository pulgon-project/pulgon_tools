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
from ase.io.vasp import read_vasp

from pulgon_tools.symmetry_projector import (
    get_adapted_matrix_withparities,
    get_linegroup_symmetry_dataset,
)
from pulgon_tools.utils import get_matrices_withPhase

GAMMA_QPOINT = 0.0
INPUT_STRUCTURE = (
    Path(__file__).resolve().parent / "data" / "poscar_WS2-10x0-MoS2-20x0.vasp"
)


def generate_primitive_gamma_symmetry_adapted_matrix(
    poscar_path: Path = INPUT_STRUCTURE,
) -> tuple[np.ndarray, list[int], dict]:
    """Generate the Gamma-point symmetry-adapted matrix for a primitive cell."""
    atom = read_vasp(poscar_path)
    (
        atom_center,
        family,
        nrot,
        a_lattice,
        ops_car_sym,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(atom)

    dict_params = {
        "qpoints": GAMMA_QPOINT,
        "nrot": nrot,
        "order": order_ops,
        "family": family,
        "a": a_lattice,
        **gen_angles,
    }
    matrices = get_matrices_withPhase(
        atom_center,
        ops_car_sym,
        GAMMA_QPOINT,
    )
    adapted, dimensions, _, _ = get_adapted_matrix_withparities(
        dict_params,
        len(atom_center),
        matrices,
    )
    summary = {
        "input": poscar_path,
        "num_atoms": len(atom_center),
        "family": family,
        "nrot": nrot,
        "a_lattice": a_lattice,
        "num_symmetry_operations": len(ops_car_sym),
    }
    return adapted, dimensions, summary


def main() -> None:
    (
        adapted,
        dimensions,
        summary,
    ) = generate_primitive_gamma_symmetry_adapted_matrix()
    orthonormal_error = np.linalg.norm(
        adapted.conj().T @ adapted - np.eye(adapted.shape[1])
    )

    print("Primitive-cell Gamma symmetry-adapted basis")
    print(f"Input structure: {summary['input']}")
    print(f"Atoms: {summary['num_atoms']}")
    print(f"Line-group family: {summary['family']}")
    print(f"Rotational symmetry nrot: {summary['nrot']}")
    print(f"Lattice period a: {summary['a_lattice']:.8f}")
    print(f"Symmetry operations: {summary['num_symmetry_operations']}")
    print(f"Adapted matrix shape: {adapted.shape}")
    print(f"Irrep block dimensions: {dimensions}")
    print(f"Sum of block dimensions: {sum(dimensions)}")
    print(f"Orthonormality error: {orthonormal_error:.3e}")


if __name__ == "__main__":
    main()
