import argparse
import ast

import numpy as np
import phonopy.file_IO
import scipy.sparse as ssp
from ase import Atoms
from ipdb import set_trace
from phonopy.harmonic.force_constants import compact_fc_to_full_fc


def main():
    parser = argparse.ArgumentParser(description="Apply the sum rules to fcs")
    parser.add_argument(
        "--path_yaml",
        default="./phonopy.yaml",
        help="The path of phonopy.yaml",
    )
    parser.add_argument(
        "--pbc",
        default=[True, True, False],
        type=parse_bool_list,
        help="The periodic boundary conduction of structure",
    )
    parser.add_argument(
        "--fcs",
        default="./force_constants.hdf5",
        help="The path of force_constants.hdf5 or FORCE_CONSTANTS",
    )
    parser.add_argument(
        "--plot_phonon",
        action="store_true",
        help="Enable plotting if specified (default: False)",
    )
    parser.add_argument(
        "--phononfig_savename",
        default="phonon_fix",
        help="The name of phonon spectrum fig",
    )
    parser.add_argument(
        "--fcs_savename",
        default="FORCE_CONSTANTS_correction.hdf5",
        help="The name of corrected fcs",
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

    path_yaml = args.path_yaml
    fcs_name = args.fcs
    pbc = args.pbc
    methods = args.methods
    plot_phonon = args.plot_phonon
    fcs_savename = args.fcs_savename
    phononfig_savename = args.phononfig_savename
    full_fcs = args.full_fcs

    phonon = phonopy.load(
        phonopy_yaml=path_yaml, force_constants_filename=fcs_name
    )
    scell = phonon.supercell
    atoms_scell = Atoms(
        numbers=scell.numbers,
        cell=scell.cell,
        positions=scell.positions,
        pbc=pbc,
    )

    IFC = phonon.force_constants
    n_atoms, n_satoms = IFC.shape[:2]

    average_delta = atoms_scell.get_all_distances(mic=True, vector=True)
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
    # positions = phonon.get_supercell().positions
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

    # And fthe Huang invariances, also for rotation.
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
        print("Start drawing the phonon spectrum")
        phonon.force_constants = IFC_sym
        phonon.auto_band_structure()
        phonon.plot_band_structure().savefig(phononfig_savename, dpi=500)

    if full_fcs:
        IFC_full = compact_fc_to_full_fc(phonon.primitive, IFC_sym)
        # phonopy.file_IO.write_FORCE_CONSTANTS(IFC_full, fcs_savename, phonon.primitive.p2s_map)
        phonopy.file_IO.write_force_constants_to_hdf5(
            IFC_full, filename=fcs_savename
        )
    else:
        phonopy.file_IO.write_force_constants_to_hdf5(
            IFC_sym, filename=fcs_savename
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


if __name__ == "__main__":
    main()
