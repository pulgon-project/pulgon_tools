import argparse
import ast
import os

import numpy as np
import phonopy.file_IO
import scipy.sparse as ssp
from ase import Atoms
from ipdb import set_trace
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, read_force_constants_hdf5
from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
    full_fc_to_compact_fc,
)
from phonopy.interface.vasp import read_vasp


def main():
    parser = argparse.ArgumentParser(description="Apply the sum rules to fcs")
    parser.add_argument(
        "--pbc",
        required=True,
        # default=[True, True, False],
        type=parse_bool_list,
        help="The periodic boundary conduction of structure",
    )
    parser.add_argument(
        "--supercell_matrix",
        required=True,
        # default=[5,5,1],
        type=parse_int_list,
        help="The supercell matrix that used to calculate fcs",
    )
    parser.add_argument(
        "-st" "--poscar",
        default="./POSCAR",
        help="The path of poscar",
    )
    parser.add_argument(
        "--path_yaml",
        default=None,
        help="The path of phonopy.yaml",
    )
    parser.add_argument(
        "--cut_off",
        default=15,
        type=float,
        help="Cutoff radius for interatomic interactions",
    )
    parser.add_argument(
        "--fcs",
        default="./FORCE_CONSTANTS",
        help="The path of force_constants.hdf5 or FORCE_CONSTANTS",
    )
    parser.add_argument(
        "--plot_phonon",
        action="store_true",
        help="Enable plotting if specified (default: False)",
    )
    parser.add_argument(
        "--k_path",
        default=None,
        type=str2list,
        help="The k path of plotting phonon, e.g. [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]",
    )
    parser.add_argument(
        "--phononfig_savename",
        default="phonon_fix",
        help="The name of phonon spectrum fig",
    )
    parser.add_argument(
        "--fcs_savename",
        default="FORCE_CONSTANTS_correction",
        help="The name of corrected fcs",
    )
    parser.add_argument(
        "--recenter",
        action="store_true",
        help="(atoms.positions - [0.5,0.5,0.5])%1",
    )
    parser.add_argument(
        "--methods",
        default="convex_opt",
        help="The available methods are 'convex_opt', 'ridge_model'",
    )
    parser.add_argument(
        "--full_fcs",
        action="store_true",
        help="Enable saving the complete fcs",
    )
    args = parser.parse_args()

    poscar = args.poscar
    supercell_matrix = args.supercell_matrix
    path_yaml = args.path_yaml
    fcs_name = args.fcs
    pbc = args.pbc
    cut_off = args.cut_off
    methods = args.methods
    plot_phonon = args.plot_phonon
    fcs_savename = args.fcs_savename
    phononfig_savename = args.phononfig_savename
    full_fcs = args.full_fcs
    recenter = args.recenter
    k_path = args.k_path

    if path_yaml is not None:
        phonon = phonopy.load(
            phonopy_yaml=path_yaml, force_constants_filename=fcs_name
        )
    else:
        unitcell = read_vasp(poscar)
        phonon = Phonopy(unitcell, supercell_matrix=np.diag(supercell_matrix))

        file_ext = os.path.splitext(fcs_name)[1].lower()
        if file_ext == ".hdf5":
            fcs = read_force_constants_hdf5(fcs_name)
        else:
            fcs = parse_FORCE_CONSTANTS(fcs_name)

        phonon.force_constants = fcs

    scell = phonon.supercell
    if recenter:
        atoms_scell = Atoms(
            numbers=scell.numbers,
            cell=scell.cell,
            scaled_positions=(scell.scaled_positions - [0.5, 0.5, 0.5]) % 1,
            pbc=pbc,
        )
    else:
        atoms_scell = Atoms(
            numbers=scell.numbers,
            cell=scell.cell,
            scaled_positions=scell.scaled_positions,
            pbc=pbc,
        )

    phonon.symmetrize_force_constants()
    IFC = phonon.force_constants.copy()
    if IFC.shape[0] == IFC.shape[1]:
        IFC = full_fc_to_compact_fc(phonon.primitive, IFC)
        n_atoms, n_satoms = IFC.shape[:2]
    else:
        n_atoms, n_satoms = IFC.shape[:2]

    average_delta = atoms_scell.get_all_distances(mic=True, vector=True)
    average_distance = atoms_scell.get_all_distances(mic=True, vector=False)

    average_products = np.einsum(
        "...i,...j->...ij", average_delta, average_delta
    )

    # %%
    n_rows = 0
    rows = []
    cols = []
    data = []
    # Acoustic sum rules for translations.
    for i in range(n_atoms):
        for alpha in range(3):
            for beta in range(3):
                for j in range(n_satoms):
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index((i, j, alpha, beta), IFC.shape)
                    )
                    data.append(1.0)
                n_rows += 1
    print("Adding acoustic sum rules")

    # The same but for rotations (Born-Huang).
    for i in range(n_atoms):
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    for j in range(n_satoms):
                        r_ij = average_delta[phonon.primitive.p2s_map[i], j]
                        rows.append(n_rows)
                        cols.append(
                            np.ravel_multi_index(
                                (i, j, alpha, beta), IFC.shape
                            )
                        )
                        data.append(r_ij[gamma])
                        rows.append(n_rows)
                        cols.append(
                            np.ravel_multi_index(
                                (i, j, alpha, gamma), IFC.shape
                            )
                        )
                        data.append(-r_ij[beta])
                    n_rows += 1
    print("Adding Born-Huang sum rules")

    # And the Huang invariances, also for rotation.
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                for delta in range(3):
                    for i in range(n_atoms):
                        for j in range(n_satoms):
                            products = average_products[
                                phonon.primitive.p2s_map[i], j
                            ]
                            rows.append(n_rows)
                            cols.append(
                                np.ravel_multi_index(
                                    (i, j, alpha, beta), IFC.shape
                                )
                            )
                            data.append(products[gamma, delta])
                            rows.append(n_rows)
                            cols.append(
                                np.ravel_multi_index(
                                    (i, j, gamma, delta), IFC.shape
                                )
                            )
                            data.append(-products[alpha, beta])
                    n_rows += 1
    print("Adding Huang invariances")

    # Make sure the IFC matrix is symmetric.
    for i in range(n_atoms):
        for j in range(n_atoms):
            for alpha in range(3):
                for beta in range(3):
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index(
                            (i, phonon.primitive.p2s_map[j], alpha, beta),
                            IFC.shape,
                        )
                    )
                    data.append(1.0)
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index(
                            (j, phonon.primitive.p2s_map[i], beta, alpha),
                            IFC.shape,
                        )
                    )
                    data.append(-1.0)
                    n_rows += 1
    print("now finish symmetric rules")

    # Add extra constraints to make the force constants short-sighted
    idx_x, idx_y = np.where(average_distance > cut_off)
    for ii, ix in enumerate(idx_x):
        if ix in phonon.primitive.p2s_map:
            i = phonon.primitive.p2p_map[ix]
            j = idx_y[ii]
            for alpha in range(3):
                for beta in range(3):
                    rows.append(n_rows)
                    cols.append(
                        np.ravel_multi_index((i, j, alpha, beta), IFC.shape)
                    )
                    data.append(1.0)
                    n_rows += 1
    print("now finish the fourth constrains")

    # Rebuild the sparse matrix.
    M = ssp.coo_array((data, (rows, cols)), shape=(n_rows, IFC.size))
    print("Start solving constraints")
    if methods == "convex_opt":
        import cvxpy as cp

        # ###############  CVXPY  #################
        flat_IFCs = IFC.ravel()
        print("M @ x  before:", np.abs(M.dot(flat_IFCs)).sum())
        x = cp.Variable(IFC.size)
        cost = cp.sum_squares(x - flat_IFCs)
        prob = cp.Problem(cp.Minimize(cost), [M @ x == 0])
        prob.solve()
        IFC_sym = x.value.reshape(IFC.shape)
        print("M @ x  after:", np.abs(M.dot(IFC_sym.ravel())).sum())
        ##########################################
    elif methods == "ridge_model":
        from sklearn.linear_model import Ridge

        ################ Ridge Model ###############
        # before fit
        parameters = IFC.ravel().copy()
        d = M.dot(parameters)
        delta = np.linalg.norm(d)
        print("Rotational sum-rules before, ||Ax|| = {:20.15f}".format(delta))
        ## fitting
        ridge = Ridge(
            alpha=1e-6, fit_intercept=False, solver="sparse_cg", tol=1e-10
        )
        ridge.fit(M, d)
        parameters -= ridge.coef_
        ## after fit
        d = M.dot(parameters)
        delta = np.linalg.norm(d)
        print("Rotational sum-rules after,  ||Ax|| = {:20.15f}".format(delta))
        IFC_sym = parameters.reshape(IFC.shape)
        ########################################################

    if plot_phonon:
        import matplotlib.pyplot as plt
        from phonopy.phonon.band_structure import (
            get_band_qpoints_and_path_connections,
        )

        print("Start drawing the phonon spectrum")
        if k_path is None:
            phonon.auto_band_structure()
            phonon.plot_band_structure().savefig(phononfig_savename, dpi=500)
        else:

            # path = [[0.0, 0.0, 0.0], [0.5, 0.0,0.0], [0.5, 0.5, 0.0],[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]
            # path = [[[0.0, 0.0, 0.0], [0.0, 0.0,0.5]]]
            # path =

            qpoints, connections = get_band_qpoints_and_path_connections(
                [k_path], npoints=101
            )

            phonon.run_band_structure(
                qpoints, path_connections=connections, with_eigenvectors=True
            )
            bands_raws = phonon.get_band_structure_dict()

            phonon.force_constants = IFC_sym
            phonon.run_band_structure(
                qpoints, path_connections=connections, with_eigenvectors=True
            )
            bands_fix = phonon.get_band_structure_dict()

            distances = bands_raws["distances"][0]
            freq_raw = bands_raws["frequencies"][0]
            freq_fix = bands_fix["frequencies"][0]

            for ii, freq in enumerate(freq_raw.T):
                if ii == 0:
                    plt.plot(distances, freq, label="raw", color="grey")
                else:
                    plt.plot(distances, freq, color="grey")

            for ii, freq in enumerate(freq_fix.T):
                if ii == 0:
                    plt.plot(distances, freq, label="fix", color="red")
                else:
                    plt.plot(distances, freq, color="red")
            plt.plot(distances, np.zeros_like(distances), "b--", linewidth=0.8)
            plt.legend()
            # plt.ylim(bottom=0)
            plt.savefig(phononfig_savename, dpi=300)

    if full_fcs:
        IFC_full = compact_fc_to_full_fc(phonon.primitive, IFC_sym)
        phonopy.file_IO.write_FORCE_CONSTANTS(
            IFC_full, fcs_savename, phonon.primitive.p2s_map
        )
        phonopy.file_IO.write_force_constants_to_hdf5(
            IFC_full, filename=fcs_savename
        )
    else:
        phonopy.file_IO.write_FORCE_CONSTANTS(
            IFC_sym, fcs_savename, phonon.primitive.p2s_map
        )
        phonopy.file_IO.write_force_constants_to_hdf5(
            IFC_sym, filename=(fcs_savename + ".hdf5")
        )


def parse_bool_list(value):
    """Parse a string representing a list of booleans into a Python list"""
    try:
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not all(
            isinstance(x, bool) for x in parsed
        ):
            raise ValueError(
                "Input must be a list of booleans, e.g., [True, True, False]"
            )
        if len(parsed) != 3:
            raise ValueError(
                "PBC list must contain exactly 3 booleans, e.g., [True, True, False]"
            )
        return parsed
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid boolean list: {value}")


def parse_int_list(value):
    """Parse a string representing a list of booleans into a Python list"""
    try:
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not all(
            isinstance(x, int) for x in parsed
        ):
            raise ValueError("Input must be a list of Int, e.g., [5, 5, 1]")
        if len(parsed) != 3:
            raise ValueError(
                "PBC list must contain exactly 3 int, e.g., [5, 5, 1]"
            )
        return parsed
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid boolean list: {value}")


def str2list(v):
    if v is None:
        return None
    try:
        return ast.literal_eval(v)
    except Exception:
        raise argparse.ArgumentTypeError(
            "k_path must be a valid Python list expression."
        )


if __name__ == "__main__":
    main()
