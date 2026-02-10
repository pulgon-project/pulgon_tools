import argparse

import numpy as np
from ase import Atom, Atoms
from ase.io import read
from ipdb import set_trace
from pymatgen.core.operations import SymmOp

from pulgon_tools.detect_generalized_translational_group import (
    CyclicGroupAnalyzer,
)
from pulgon_tools.detect_point_group import LineGroupAnalyzer
from pulgon_tools.line_group_table import get_family_Num_from_sym_symbol
from pulgon_tools.utils import (
    brute_force_generate_group_subsquent,
    find_axis_center_of_nanotube,
    get_character_num_withparities,
    get_character_withparities,
)


def get_linegroup_symmetry_dataset(poscar):
    if type(poscar) == str:
        atom = read(poscar)
    elif type(poscar) == Atom or type(poscar) == Atoms:
        atom = poscar
    else:
        print("Unknown input")

    atom_center = find_axis_center_of_nanotube(atom)

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)

    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol

    trans_op = np.round(cyclic.get_generators(), 6)
    rots_op = np.round(obj.get_generators(), 6)
    mats = np.vstack(([trans_op], rots_op))
    ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-2)

    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    return atom_center, family, nrot, aL, ops_car_sym, order_ops


def main():
    parser = argparse.ArgumentParser(
        description="Return the representation matrices or character table from a structure"
    )
    parser.add_argument("filename", help="path to the file of a strcuture")
    parser.add_argument(
        "-qp",
        "--qpoint_z",
        type=float,
        default=0.0,
        help="The qpoint in the periodic direction (z), from 0 to 1",
    )
    parser.add_argument(
        "-symprec",
        "--symmetry_precision",
        type=float,
        default=1e-8,
        help="the symmetry tolerance",
    )
    parser.add_argument(
        "-charaname",
        "--character_savename",
        default="characters",
        help="The filename of character",
    )
    parser.add_argument(
        "--enable_rep_matrix",
        action="store_true",
        help="open the detection of point group",
    )

    args = parser.parse_args()
    st_name = args.filename
    qpoint_z = args.qpoint_z
    symprec = args.symmetry_precision
    chara_filename = args.character_savename
    enable_rep_matrix = args.enable_rep_matrix

    atom = read(st_name)
    (
        atom_center,
        family,
        nrot,
        aL,
        ops_car_sym,
        order_ops,
    ) = get_linegroup_symmetry_dataset(atom)
    qp_normalized = qpoint_z / aL * 2 * np.pi
    DictParams = {
        "qpoints": qp_normalized,
        "nrot": nrot,
        "order": order_ops,
        "family": family,
        "a": aL,
    }

    if enable_rep_matrix:
        (
            characters,
            ireps_values,
            ireps_symbols,
        ) = get_character_num_withparities(DictParams, symprec=symprec)
        representation_mat, _, _ = get_character_withparities(
            DictParams, symprec=symprec
        )
        representation_mat_dict = {}
        for i, rep in enumerate(representation_mat):
            representation_mat_dict[f"D_irrep_{i}"] = rep

        np.savez(
            chara_filename,
            characters=characters,
            ireps_values=ireps_values,
            ireps_symbols=ireps_symbols,
            **representation_mat_dict,
        )

    else:
        (
            characters,
            ireps_values,
            ireps_symbols,
        ) = get_character_num_withparities(DictParams, symprec=symprec)
        np.savez(
            chara_filename,
            characters=characters,
            ireps_values=ireps_values,
            ireps_symbols=ireps_symbols,
        )


if __name__ == "__main__":
    main()
