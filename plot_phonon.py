import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pytest
import seaborn as sns
from ase import Atoms
from ase.geometry import distance
from ipdb import set_trace
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from tqdm import tqdm

from pulgon_tools.symmetry_projector import (
    get_adapted_matrix_withparities,
    get_eigenmodes_from_phonon,
    get_linegroup_symmetry_dataset,
)
from pulgon_tools.utils import get_matrices_withPhase

SP_DATA = "test/data/symmetry_projector"
path_save_phonon = "phonon.png"


def phonon_obj():
    """Load the phonopy object from symmetry_projector test data."""
    return phonopy.load(
        phonopy_yaml=f"{SP_DATA}/phonopy.yaml",
        force_constants_filename=f"{SP_DATA}/FORCE_CONSTANTS",
    )


def symmetry_dataset(phonon_obj):
    """Build symmetry dataset from phonon primitive cell."""
    prim = phonon_obj.primitive
    atom = Atoms(
        cell=prim.cell, numbers=prim.numbers, positions=prim.positions
    )
    (
        atom_center,
        family,
        nrot,
        aL,
        ops_car_sym,
        order_ops,
        gen_angles,
    ) = get_linegroup_symmetry_dataset(atom)
    return {
        "atom_center": atom_center,
        "family": family,
        "nrot": nrot,
        "aL": aL,
        "ops_car_sym": ops_car_sym,
        "order_ops": order_ops,
        "gen_angles": gen_angles,
        "num_atom": len(prim.numbers),
    }


def adapted_result_gamma(symmetry_dataset):
    """Run get_adapted_matrix_withparities at Gamma point."""
    ds = symmetry_dataset
    qp_1dim = 0.0
    DictParams = {
        "qpoints": qp_1dim,
        "nrot": ds["nrot"],
        "order": ds["order_ops"],
        "family": ds["family"],
        "a": ds["aL"],
        **ds["gen_angles"],
    }
    matrices = get_matrices_withPhase(
        ds["atom_center"], ds["ops_car_sym"], qp_1dim, symprec=1e-3
    )
    (
        adapted,
        dimensions,
        paras_values,
        paras_symbols,
    ) = get_adapted_matrix_withparities(DictParams, ds["num_atom"], matrices)
    return adapted, dimensions, paras_values, paras_symbols, ds


def main():
    phonon = phonon_obj()
    NQS = 51
    path = [[[0, 0, -0.5], [0, 0, 0.5]]]
    q_vectors, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    q_vectors = q_vectors[0]

    freqs_raw, freqs_adp = [], []
    dimensions_tol, paras_values_tol, paras_symbols_tol = [], [], []
    for ii, q_vector in enumerate(tqdm(q_vectors)):
        freq_raw, _, _ = get_eigenmodes_from_phonon(
            phonon, q_vector, adapted=False
        )
        (
            freq_adp,
            _,
            _,
            dimensions,
            paras_values,
            paras_symbols,
        ) = get_eigenmodes_from_phonon(phonon, q_vector, adapted=True)
        dimensions_tol.append(dimensions)
        paras_values_tol.append(paras_values)
        paras_symbols_tol.append(paras_symbols)

        freqs_raw.append(freq_raw)
        freqs_adp.append(freq_adp)
    freqs_raw = np.array(freqs_raw).T  # * 2 * np.pi
    freqs_adp = np.array(freqs_adp).T  # * 2 * np.pi

    distances = q_vectors[:, -1]

    # Common settings
    sym_names = [str(s) for s in paras_symbols_tol[0]]
    m_idx = sym_names.index("m1") if "m1" in sym_names else 1
    labelsize = 14
    fontsize = 16

    # Helper: plot raw phonon on an axis
    def plot_raw(ax):
        for i, f_raw in enumerate(freqs_raw):
            if i == 0:
                ax.plot(
                    distances,
                    f_raw,
                    color="grey",
                    label="raw",
                    zorder=1,
                )
            else:
                ax.plot(distances, f_raw, color="grey", zorder=1)

    # ===================== Figure 1: colored by m =====================
    fig1, ax1 = plt.subplots()
    plot_raw(ax1)

    all_m_values = sorted(
        set(
            int(abs(paras_values_tol[ii][idx_ir][m_idx]))
            for ii in range(len(q_vectors))
            for idx_ir in range(len(dimensions_tol[ii]))
        )
    )
    cmap_m = plt.cm.tab10
    m_color_map = {m: cmap_m(i % 10) for i, m in enumerate(all_m_values)}

    plotted_labels = set()
    for ii, freq in enumerate(freqs_adp.T):
        dim_sum = np.cumsum(dimensions_tol[ii])
        for jj in range(len(freq)):
            idx_ir = np.searchsorted(dim_sum, jj, side="right")
            m_val = int(abs(paras_values_tol[ii][idx_ir][m_idx]))
            label = f"m={m_val}"
            color = m_color_map[m_val]
            if label not in plotted_labels:
                ax1.scatter(
                    distances[ii],
                    freq[jj],
                    color=color,
                    label=label,
                    s=8,
                    zorder=2,
                )
                plotted_labels.add(label)
            else:
                ax1.scatter(
                    distances[ii],
                    freq[jj],
                    color=color,
                    s=8,
                    zorder=2,
                )

    ax1.set_xlabel("q", fontsize=fontsize)
    ax1.set_ylabel(r"$\omega$" + " (THz)", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.legend(fontsize=10, loc="upper left")
    fig1.tight_layout()
    fig1.savefig("phonon_by_m.png", dpi=300)

    # ============ Figure 2: colored by full irrep label ============
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_raw(ax2)

    # Build full irrep labels: m=X, PiU=Y, PiV=Z, PiH=W
    # Find indices for parity symbols
    parity_keys = ["piU", "piV", "piH"]
    parity_indices = []
    for pk in parity_keys:
        if pk in sym_names:
            parity_indices.append(sym_names.index(pk))
        else:
            parity_indices.append(None)

    def make_full_label(val):
        m_val = int(abs(val[m_idx]))
        parts = [f"m={m_val}"]
        for pk, pi in zip(parity_keys, parity_indices):
            if pi is not None:
                pname = pk.replace("pi", r"\Pi_")
                parts.append(f"${pname}{{{int(val[pi])}}}$")
        return ", ".join(parts)

    all_full_labels = []
    for ii in range(len(q_vectors)):
        for idx_ir in range(len(dimensions_tol[ii])):
            label = make_full_label(paras_values_tol[ii][idx_ir])
            if label not in all_full_labels:
                all_full_labels.append(label)

    # Generate N distinct, high-contrast colors
    # Combine tab10 + Set1 + Dark2 + tab20 bold halves
    _palettes = (
        # list(plt.cm.tab50.colors)      # 10 colors
        list(sns.color_palette("husl", 25))
        # + list(plt.cm.Set1.colors)      # 9 colors
        # + list(plt.cm.Dark2.colors)     # 8 colors
    )
    # Deduplicate while preserving order
    seen = set()
    distinct_colors = []
    for c in _palettes:
        key = tuple(round(x, 3) for x in c)
        if key not in seen:
            seen.add(key)
            distinct_colors.append(c)
    full_color_map = {
        lbl: distinct_colors[i % len(distinct_colors)]
        for i, lbl in enumerate(all_full_labels)
    }

    plotted_labels = set()
    for ii, freq in enumerate(freqs_adp.T):
        dim_sum = np.cumsum(dimensions_tol[ii])
        for jj in range(len(freq)):
            idx_ir = np.searchsorted(dim_sum, jj, side="right")
            label = make_full_label(paras_values_tol[ii][idx_ir])
            color = full_color_map[label]
            if label not in plotted_labels:
                ax2.scatter(
                    distances[ii],
                    freq[jj],
                    color=color,
                    label=label,
                    s=8,
                    zorder=2,
                )
                plotted_labels.add(label)
            else:
                ax2.scatter(
                    distances[ii],
                    freq[jj],
                    color=color,
                    s=8,
                    zorder=2,
                )

    ax2.set_xlabel("q", fontsize=fontsize)
    ax2.set_ylabel(r"$\omega$" + " (THz)", fontsize=fontsize)
    ax2.tick_params(labelsize=labelsize)
    ax2.legend(
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    fig2.tight_layout()
    fig2.savefig("phonon_by_irrep.png", dpi=300)


if __name__ == "__main__":
    main()
