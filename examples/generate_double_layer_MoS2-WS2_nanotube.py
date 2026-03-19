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


import numpy as np
from ase import Atoms
from ase.io.vasp import read_vasp, write_vasp

from pulgon_tools.structure_chirality import get_nanotube_from_n1n2

if __name__ == "__main__":
    ###################### inner-layer ######################
    n1, n2 = 10, 0
    symbol1, symbol2 = 74, 16  # W, S
    name1 = "WS2"
    atom1 = read_vasp("examples/poscar_monolayer_%s" % name1)
    delta_Z1 = abs((atom1.positions - atom1.positions[2])[0, 2])
    bond_length1 = np.linalg.norm((atom1.positions[0] - atom1.positions[2]))

    new_atom1 = get_nanotube_from_n1n2(
        n1, n2, symbol1, symbol2, bond_length1, delta_Z1
    )

    ###################### outer-layer ######################

    n3, n4 = 20, 0
    symbol1, symbol2 = 42, 16  # Mo, S
    name2 = "MoS2"
    atom2 = read_vasp("examples/poscar_monolayer_%s" % name2)
    delta_Z2 = abs((atom2.positions - atom2.positions[2])[0, 2])
    bond_length2 = np.linalg.norm((atom2.positions[0] - atom2.positions[2]))

    new_atom2 = get_nanotube_from_n1n2(
        n3, n4, symbol1, symbol2, bond_length2, delta_Z2
    )
    name = "WS2-%dx%d-MoS2-%dx%d" % (n1, n2, n3, n4)

    pos1 = (
        (new_atom1.get_scaled_positions() - [0.5, 0.5, 0]) % 1 @ new_atom1.cell
    )
    pos2 = (
        (new_atom2.get_scaled_positions() - [0.5, 0.5, 0]) % 1 @ new_atom2.cell
    )

    tmp = np.concatenate(
        ((new_atom2.cell - new_atom1.cell)[[0, 1], [0, 1]] / 2, [0])
    )
    pos1 = pos1 + tmp

    pos = np.vstack((pos1, pos2))
    cell = new_atom2.cell
    numbers = np.concatenate((new_atom1.numbers, new_atom2.numbers), axis=0)
    new_atom = Atoms(positions=pos, cell=cell, numbers=numbers)
    write_vasp("poscar_%s.vasp" % (name), new_atom, direct=True, sort=True)
