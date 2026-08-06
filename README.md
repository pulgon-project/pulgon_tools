[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19223341.svg)](https://doi.org/10.5281/zenodo.19223341)
[![PyPI version](https://badge.fury.io/py/pulgon-tools.svg)](https://badge.fury.io/py/pulgon-tools)
[![Build with uv and test](https://github.com/pulgon-project/pulgon_tools/actions/workflows/test-with-uv.yaml/badge.svg)](https://github.com/pulgon-project/pulgon_tools/actions/workflows/test-with-uv.yaml)
[![Coverage Status](https://coveralls.io/repos/github/pulgon-project/pulgon_tools/badge.svg?branch=main)](https://coveralls.io/github/pulgon-project/pulgon_tools?branch=main)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# pulgon_tools

This repository contains the software backbone of [project PULGON](https://pulgon-project.org/), focused on the study of quasi-1D nanostructures such as nanotubes and nanowires from the point of view of line groups, which describe their symmetries.

---

## Installation

### With uv:

```bash
git clone https://github.com/pulgon-project/pulgon_tools.git
cd pulgon_tools
uv sync --extra dev
uv run pytest                    # run the test suite
```

### With pip:

```bash
git clone https://github.com/pulgon-project/pulgon_tools.git
cd pulgon_tools
pip install -e ".[dev]"
pytest                    # run the test suite
```

**Requirements:** Python ≥ 3.9, ≤ 3.12

---

## Modules Overview

```
Input motif / Generators / Chiral indices
        │
        ▼
┌───────────────────────┐
│  Structure Generation │  → POSCAR / cif / xyz
└───────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Symmetry Detection  │  → Line group symbol, symmetry operations
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Irreps & Char. Table │  → Irrep matrices, character table
└──────────────────────┘

Structure + Supercell + IFC
        │
        ▼
┌───────────────────┐
│   IFC Correction  │  → Modified IFC
└───────────────────┘
```

---

## Module 1: Structure Generation

Generates quasi-1D structures from line-group symmetry. Two approaches are supported.

### 1a. Symmetry-based approach

Builds a periodic structure from a set of line-group generators and an atomic motif in cylindrical coordinates `(r, φ, z)`.

```bash
pulgon-generate-structures-sym_based \
  -m R1 PHI1 Z1 \
     R2 PHI2 Z2 \
  -b SYMBOL1 SYMBOL2 \
  -g APG_GENERATOR  \
  -c TG_TYPE TG_VALUE \
  -s OUTPUT_FILENAME
```

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--motif` | Motif coordinates as `r φ z` groups; the number of values must be a multiple of three | `3 0 0 2.2 0.2618 0` |
| `-b`, `--symbol` | Atomic species symbols in the same order as the `-m` motif atoms | `C N` |
| `-g`, `--generators` | Axial point group generators. Supported: `Cn(number)`, `S2n(number)`, `U_d(angle)`, `sigmaV()`, `sigmaH()`, `U()` | `Cn(8)` |
| `-c`, `--cyclic` | Generalized translational group: `T_Q Q f` for screw, or `T_V f` for glide; `f` is the z translation in Å | `T_Q 3 1.6` |
| `-s`, `--st_name` | Output filename | `structure.vasp` |

**Example** — C/N nanotube with C₈ symmetry and screw translation T₃(1.6):

```bash
pulgon-generate-structures-sym_based \
  -m 3 0 0 \
     2.2 0.2618 0 \
  -b C N \
  -g "Cn(8)" \
  -c T_Q 3 1.6 \
  -s structure.vasp
```

### 1b. Chiral roll-up (MoS₂-type nanotubes)

Generates MX₂-type nanotubes from chiral indices `(n, m)` by rolling up a hexagonal 2D parent lattice.

```bash
pulgon-generate-structures-chirality \
  -c N M \
  -b METAL CHALCOGEN \
  -l BOND_LENGTH \
  -d DELTA_Z \
  -s OUTPUT_FILENAME
```

| Flag                  | Description                                                                 | Default       |
| --------------------- | --------------------------------------------------------------------------- | ------------- |
| `-c`, `--chirality`   | Chiral indices as two non-negative integers`n m`                            | `(10, 10)`    |
| `-b`, `--symbol`      | Exactly two atomic symbols: metal and chalcogen                             | `('Mo', 'S')` |
| `-l`, `--bond_length` | M–X bond length in Å                                                      | `2.43`        |
| `-d`, `--delta_Z`     | Pre-roll-up layer spacing between the M and X layers in the 2D sheet, in Å | `1.57`        |
| `-s`, `--st_name`     | Output filename                                                             | `POSCAR`      |

**Example** — MoS₂ zigzag, armchair, and chiral nanotubes:

```bash
pulgon-generate-structures-chirality -c 8 0 -b Mo S -l 2.43 -s mos2_zigzag.vasp
pulgon-generate-structures-chirality -c 8 8 -b Mo S -l 2.43 -s mos2_armchair.vasp
pulgon-generate-structures-chirality -c 8 4 -b Mo S -l 2.43 -s mos2_chiral.vasp
```

---

## Module 2: Symmetry Detection

Identifies the complete line-group symmetry of a quasi-1D structure. The detection proceeds in two stages: generalized translational group (screw / glide) detection, followed by axial point group classification.

Two tolerance settings are used during symmetry detection:

- `--tolerance` is the atomic-coordinate matching tolerance. It controls whether transformed atoms are considered equivalent and is used by both cyclic-group and axial-point-group detection.
- `--layer-tolerance` is only used by cyclic-group detection. It is a fractional tolerance along the periodic z direction for grouping layers into monomer translation candidates; it does not replace the coordinate-matching tolerance.

### Detect line group

```bash
pulgon-detect-linegroup -p POSCAR [OPTIONS]
```

| Flag                      | Description                                                               | Default |
| ------------------------- | ------------------------------------------------------------------------- | ------- |
| `-p`, `--POSCAR`          | Input structure file in a format readable by ASE                          | —      |
| `-t`, `--tolerance`       | Distance tolerance in Å for matching transformed atoms to existing atoms | `0.01`  |
| `-d`, `--layer-tolerance` | Fractional z-layer tolerance for cyclic monomer translation candidates    | `0.05`  |

This command reports the generalized translational group `Z`, axial point group `P`, and line-group family number.

```bash
pulgon-detect-linegroup -p POSCAR -t 1e-2
```

### Detect generalized translational group

```bash
pulgon-detect-CyclicGroup -p POSCAR [OPTIONS]
```

| Flag                      | Description                                                     | Default |
| ------------------------- | --------------------------------------------------------------- | ------- |
| `-p`, `--POSCAR`          | Input structure file in a format readable by ASE                | —      |
| `-t`, `--tolerance`       | Distance tolerance in Å for cyclic-group atom matching         | `0.01`  |
| `-d`, `--layer-tolerance` | Fractional z-layer tolerance for monomer translation candidates | `0.05`  |
| `-o`, `--enable_log`      | Print detailed debug logs for cyclic-group detection            | `False` |

### Detect axial point group

```bash
pulgon-detect-AxialPointGroup -p POSCAR [OPTIONS]
```

| Flag                 | Description                                                  | Default |
| -------------------- | ------------------------------------------------------------ | ------- |
| `-p`, `--POSCAR`     | Input structure file in a format readable by ASE             | —      |
| `-t`, `--tolerance`  | Distance tolerance in Å for axial point-group atom matching | `0.01`  |
| `-g`, `--enable_pg`  | Also print pymatgen's full molecular point-group result      | `False` |
| `-o`, `--enable_log` | Print detailed debug logs for axial point-group detection    | `False` |

**Example** — detect the line group of a (5,5) SWCNT:

```bash
pulgon-detect-linegroup -p POSCAR
pulgon-detect-CyclicGroup -p POSCAR
pulgon-detect-AxialPointGroup -p POSCAR
```

The 13 line-group families are classified by the combination of generalized translational group Z (pure translation T, screw T_Q, or glide T_C) and axial point group P. See Table 1 in the paper for a complete listing.

---

## Module 3: Irreps and Character Table

Constructs irreducible representations and character tables for all 13 line-group families, following the framework of Damnjanović and Milošević. Each irrep is labeled by physically meaningful quantum numbers: axial wave vector k, angular momentum index m, and parity labels Π.

```bash
pulgon-irreps-tables -p POSCAR [OPTIONS]
```

| Flag                         | Description                                                                                            | Default      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | ------------ |
| `-p`, `--POSCAR`             | Input structure file in a format readable by ASE                                                       | —           |
| `-q`, `--qpoint_z`           | Reduced q coordinate along periodic z; internally converted to `qpoint_z * 2π/a`                       | `0.0`        |
| `-t`, `--tolerance`          | Distance tolerance in Å for line-group symmetry detection                                              | `0.01`       |
| `-u`, `--qpoint-tolerance`   | Numerical tolerance for special q points and boundary angular-momentum sectors in irrep construction   | `1e-6`       |
| `-s`, `--savename_chara`     | Output base filename for the `.npz` file                                                               | `characters` |
| `-r`, `--enable_rep_matrix`  | Also save irreducible representation matrices as `D_irrep_*` arrays                                    | `False`      |

Here `--tolerance` acts on Cartesian atomic-position matching during symmetry detection, while `--qpoint-tolerance` is a dimensionless numerical tolerance used in the irrep/character-table stage.

**Example** — character table of MoS₂-(5,0) at q = 0:

```bash
pulgon-irreps-tables -p POSCAR -q 0.0 -s mos2_chartable
```

The output is saved by `numpy.savez`; for example, `-s mos2_chartable` writes `mos2_chartable.npz`.

### Saved `.npz` fields

Let `N_irrep` be the number of irreducible representations, `N_op` the number of finite-factor-group operations, and `N_class` the number of ordinary conjugacy classes. The following fields are always saved:

| Field | Shape | Description |
| ----- | ----- | ----------- |
| `characters` | `(N_irrep, N_op)` | Character table with one column per operation. The original column order is preserved. |
| `ireps_values` | irrep dependent | Numerical quantum-number values associated with the irrep rows. This legacy key retains its original spelling. |
| `ireps_symbols` | irrep dependent | Symbols used to interpret the entries in `ireps_values`. |
| `qpoint_z` | scalar | Reduced axial q coordinate supplied with `--qpoint_z`. |
| `operation_labels` | `(N_op,)` | Human-readable affine-operation labels, including axes or plane normals and any fractional translation. |
| `operation_words` | `(N_op,)` | Comma-separated generator words reproducing the operation order. `0` denotes the identity seed and positive integers are one-based generator indices. |
| `conjugacy_class_scope` | scalar string | Currently `finite_factor_group`; it identifies the group whose ordinary conjugacy classes are reported. |
| `conjugacy_class_ids` | `(N_op,)` | Zero-based class ID for every operation/character-table column. |
| `operation_class_labels` | `(N_op,)` | Class label repeated in character-table column order for convenient reporting. |
| `conjugacy_class_labels` | `(N_class,)` | Labels formed from each class size and a representative operation, for example `2 S4(+90deg,axis=z)`. |
| `conjugacy_class_sizes` | `(N_class,)` | Number of operations in each conjugacy class. |
| `conjugacy_class_representatives` | `(N_class,)` | Column index selected as the representative of each class. |
| `conjugacy_class_members` | `(N_class, max_class_size)` | Column indices in each class, padded with `-1`. |
| `class_characters_available` | scalar bool | Whether an ordinary class-compressed character table is valid and included. |
| `class_characters_scope` | scalar string | `finite_factor_group_at_gamma` or `unavailable_non_gamma`. |
| `class_characters` | `(N_irrep, N_class)` or `(N_irrep, 0)` | One character column per ordinary conjugacy class at a Gamma-equivalent q point; empty away from Gamma. |
| `q_preserving_mask` | `(N_op,)` | Boolean mask selecting operations that leave q invariant modulo a reciprocal lattice vector. |
| `little_group_operation_indices` | `(N_little_op,)` | Original character-table column indices belonging to the q-point little group. |
| `little_group_conjugacy_class_scope` | scalar string | `q_preserving_subgroup_of_finite_factor_group`; identifies the subgroup used for the following class fields. |
| `little_group_conjugacy_class_ids` | `(N_op,)` | Little-group class IDs in original column order; nonmembers are `-1`. |
| `little_group_conjugacy_class_labels` | `(N_little_class,)` | Labels for ordinary conjugacy classes of the q-preserving subgroup. |
| `little_group_conjugacy_class_sizes` | `(N_little_class,)` | Sizes of the little-group conjugacy classes. |
| `little_group_conjugacy_class_representatives` | `(N_little_class,)` | Original column index representing each little-group class. |
| `little_group_conjugacy_class_members` | `(N_little_class, max_little_class_size)` | Original column indices in each little-group class, padded with `-1`. |

With `--enable_rep_matrix`, the file additionally contains `D_irrep_0`, `D_irrep_1`, ... with the representation matrices for the corresponding irrep rows.

At Gamma, the finite-factor-group characters are checked to be constant within every conjugacy class before `class_characters` is saved. Away from Gamma, nonsymmorphic Bloch phases can produce projective little-group representations, so the program does not compress `characters` using ordinary factor-group classes. It instead saves an empty `class_characters` array together with `q_preserving_mask` and the separately labeled ordinary classes of the q-preserving subgroup.

The labels are deterministic, coordinate-explicit descriptions of the computed operations rather than manually assigned conventional character-table symbols. For example, `sigma_v(normal=y) | z=1/2` denotes a vertical mirror with normal along y followed by a half-cell axial translation.

The new metadata fields use numeric and Unicode NumPy arrays rather than object arrays, so they can be accessed without enabling pickle. The legacy `ireps_symbols` field remains an object array for backward compatibility.

```python
import numpy as np

with np.load("mos2_chartable.npz", allow_pickle=False) as table:
    for column, (operation, conjugacy_class) in enumerate(
        zip(table["operation_labels"], table["operation_class_labels"])
    ):
        print(column, operation, conjugacy_class)
```

---

## Module 4: IFC Correction

Enforces invariance sum rules on second-order harmonic interatomic force constants (IFCs) of quasi-1D systems, including acoustic, Born-Huang rotational, Huang, matrix-symmetry, and cutoff constraints. Formulated as a linearly constrained quadratic optimization problem solved via CVXPY/OSQP.

```bash
pulgon-fcs-correction -p POSCAR -x SUPERCELL_MATRIX -f FORCE_CONSTANTS [OPTIONS]
```

| Flag                       | Description                                                                                               | Default             |
| -------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------- |
| `-p`, `--POSCAR`           | Input structure file used when`--path_yaml` is not provided                                               | `POSCAR`            |
| `-x`, `--supercell_matrix` | Diagonal supercell size used for the force constants, e.g.`"[1,1,5]"`                                     | —                  |
| `-y`, `--path_yaml`        | Optional`phonopy.yaml`; when provided, phonopy loads structure and supercell settings from this file      | —                  |
| `-f`, `--fcs`              | Input force constants file, either`FORCE_CONSTANTS` or `force_constants.hdf5`                             | `./FORCE_CONSTANTS` |
| `-n`, `--plot_phonon`      | Plot phonon bands before and after correction and save`phonon_fix.png`                                    | `False`             |
| `-k`, `--k_path`           | Band path used with`--plot_phonon`, e.g. `"[[0,0,0],[0.5,0,0],[0,0,0]]"`                                  | —                  |
| `-r`, `--recenter`         | Recenter fractional coordinates using`(scaled_positions - [0.5,0.5,0.5]) % 1` before building constraints | `False`             |
| `-m`, `--methods`          | Solver used to enforce constraints:`convex_opt` or `ridge_model`                                          | `convex_opt`        |
| `-z`, `--full_fcs`         | Write full supercell force-constant matrix instead of compact primitive-to-supercell form                 | `False`             |

**Example** — correct IFCs for a (12,12) MoS₂ nanotube:

```bash
pulgon-fcs-correction \
  -p POSCAR \
  -x "[1,1,5]" \
  -f FORCE_CONSTANTS \
  -n \
  -m convex_opt
```

### Python API

All modules expose a Python API for integration into custom workflows:

```python
import numpy as np
import phonopy
from phonopy.interface.vasp import read_vasp
from pulgon_tools.force_constant_correction import (
    build_constraint_matrix, solve_fcs
)

# Step 1: Load structure with Phonopy
unitcell = read_vasp("POSCAR")
phonon = phonopy.Phonopy(unitcell, supercell_matrix=np.diag([1, 1, 5]))
phonon.force_constants = phonopy.file_IO.parse_FORCE_CONSTANTS(
    "FORCE_CONSTANTS"
)

# Step 2: Build the sparse constraint matrix that enforces all constraints
M, IFC = build_constraint_matrix(phonon, recenter=False)

# Step 3: Solve the constrained quadratic optimization problem
IFC_corrected = solve_fcs(IFC, M, methods="convex_opt")

# Step 4: Save the corrected IFCs
phonopy.file_IO.write_force_constants_to_hdf5(
    IFC_corrected, filename="FORCE_CONSTANTS_correction.hdf5"
)
```

Additional examples for all modules are available in the `examples/` directory of the repository.

---

## Citation

If you use Pulgon-tools in your research, please cite:
