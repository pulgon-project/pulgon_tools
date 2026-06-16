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
  -m MOTIF \
  -b SYMBOL \
  -g APG_GENERATOR \
  -c TG_GENERATOR \
  -s OUTPUT_FILENAME
```

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--motif` | Cylindrical coordinates `[r, φ, z]` of the initial atomic motif as a Python literal list. Use numeric values only, not expressions such as `np.pi/24`. | `[[3, π/24, 0.6], [2.2, π/24, 0.8]]` |
| `-b`, `--symbol` | Atomic species symbols as a Python literal tuple/list of strings, e.g. `"('Mo', 'S')"` | `('Mo', 'S')` |
| `-g`, `--generators` | Axial point group generators as a Python literal list of whitelisted strings. Supported: `Cn(number)`, `S2n(number)`, `U_d(angle)`, `sigmaV()`, `sigmaH()`, `U()` | `['Cn(6)', 'sigmaV()']` |
| `-c`, `--cyclic` | Generalized translational group as a Python literal dict. Supported keys: `T_Q: [Q, f]` for screw (positive rotation order Q and translation f Å); `T_V: f` for glide (positive translation f Å). | `{'T_Q': [6, 1.5]}` |
| `-s`, `--st_name` | Output filename | `poscar.vasp` |

**Example** — C/N nanotube with C₈ symmetry and screw translation T₃(1.6):

```bash
pulgon-generate-structures-sym_based \
  -m "[[3, 0, 0], [2.2, 0.2618, 0]]" \
  -b "('C', 'N')" \
  -g "['Cn(8)']" \
  -c "{'T_Q': [3, 1.6]}" \
  -s structure.vasp
```

### 1b. Chiral roll-up (MoS₂-type nanotubes)

Generates MX₂-type nanotubes from chiral indices `(n, m)` by rolling up a hexagonal 2D parent lattice.

```bash
pulgon-generate-structures-chirality \
  -c CHIRALITY \
  -b SYMBOLS \
  -l BOND_LENGTH \
  -d DELTA_Z \
  -s OUTPUT_FILENAME
```

| Flag | Description | Default |
|------|-------------|---------|
| `-c`, `--chirality` | Chiral indices as a Python literal pair of non-negative integers, e.g. `"(8, 4)"` | — |
| `-b`, `--symbol` | Atomic species as a Python literal pair of strings, e.g. `"('Mo', 'S')"` | `('Mo', 'S')` |
| `-l`, `--bond_length` | M–X bond length in Å | `2.43` |
| `-d`, `--delta_Z` | Vertical offset dz of X atoms from the M layer in Å | `1.57` |
| `-s`, `--st_name` | Output filename | `POSCAR` |

**Example** — MoS₂ zigzag, armchair, and chiral nanotubes:

```bash
pulgon-generate-structures-chirality -c "(8,0)" -b "('Mo','S')" -l 2.43 -s mos2_zigzag.vasp
pulgon-generate-structures-chirality -c "(8,8)" -b "('Mo','S')" -l 2.43 -s mos2_armchair.vasp
pulgon-generate-structures-chirality -c "(8,4)" -b "('Mo','S')" -l 2.43 -s mos2_chiral.vasp
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

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--POSCAR` | Path to input POSCAR structure | — |
| `-t`, `--tolerance` | Atomic-coordinate matching tolerance for symmetry detection | `0.01` |
| `-d`, `--layer-tolerance` | Fractional z-layer tolerance for cyclic monomer translation candidates | `0.05` |

This command reports the generalized translational group `Z`, axial point group `P`, and line-group family number.

```bash
pulgon-detect-linegroup -p POSCAR -t 1e-2
```

### Detect generalized translational group

```bash
pulgon-detect-CyclicGroup -p POSCAR [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--poscar` | Path to input structure (POSCAR/cif/xyz) | — |
| `-t`, `--tolerance` | Atomic-coordinate matching tolerance for cyclic symmetry detection | `0.01` |
| `-d`, `--layer-tolerance` | Fractional z-layer tolerance for monomer translation candidates | `0.05` |
| `-o`, `--output` | Enable logging to file | `False` |

### Detect axial point group

```bash
pulgon-detect-AxialPointGroup -p POSCAR [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--poscar` | Path to input structure | — |
| `-t`, `--tolerance` | Atomic-coordinate matching tolerance for axial point-group detection | `0.01` |
| `-g`, `--group` | Enable full point group detection | `False` |
| `-o`, `--output` | Enable logging to file | `False` |

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

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--poscar` | Path to input structure | — |
| `-q`, `--qpoint` | Axial wave vector k (normalized, 0 to 1) | `0.0` |
| `-t`, `--tolerance` | Numerical tolerance | `0.01` |
| `-r`, `--representations` | Save full representation matrices to file | `False` |
| `-s`, `--st_name` | Output filename for the character table | — |

**Example** — character table of MoS₂-(5,0) at k = 0:

```bash
pulgon-irreps-tables -p POSCAR -q 0.0 -s mos2_chartable.npz
```

---

## Module 4: IFC Correction

Enforces translational and rotational invariance conditions (Born–Huang sum rules, Huang invariance conditions) on second-order harmonic interatomic force constants (IFCs). Formulated as a linearly constrained quadratic optimization problem solved via CVXPY/OSQP.

```bash
pulgon-fcs-correction -p POSCAR -x SUPERCELL_MATRIX -f FORCE_CONSTANTS [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--POSCAR` | Path to primitive structure | `POSCAR` |
| `-x`, `--supercell_matrix` | Supercell matrix, e.g. `"[1,1,3]"` | — |
| `-y`, `--path_yaml` | Path to `phonopy.yaml`; if provided, load structure and force constants from phonopy | — |
| `-f`, `--fcs` | Path to `force_constants.hdf5` or `FORCE_CONSTANTS` | `./FORCE_CONSTANTS` |
| `-n`, `--plot_phonon` | Plot phonon spectrum before and after correction | `False` |
| `-k`, `--k_path` | k-path for phonon plotting, e.g. `"[[0,0,0],[0.5,0,0]]"` | — |
| `-r`, `--recenter` | Recenter supercell scaled positions before building constraints | `False` |
| `-m`, `--methods` | Optimization backend: `convex_opt` or `ridge_model` | `convex_opt` |
| `-z`, `--full_fcs` | Save full force constants instead of compact force constants | `False` |

**Example** — correct IFCs for a (12,12) MoS₂ nanotube:

```bash
pulgon-fcs-correction \
  -p POSCAR \
  -x "[1,1,3]" \
  -f FORCE_CONSTANTS \
  -n \
  -m convex_opt
```

### Python API

All modules expose a Python API for integration into custom workflows:

```python
import phonopy
from phonopy.interface.vasp import read_vasp
from pulgon_tools.force_constant_correction import (
    build_constraint_matrix, solve_fcs
)

# Step 1: Load structure with Phonopy
unitcell = read_vasp("POSCAR")
phonon = phonopy.Phonopy(unitcell, supercell_matrix=np.diag([1, 1, 3]))
phonon.force_constants = phonopy.file_IO.parse_FORCE_CONSTANTS(
    "FORCE_CONSTANTS"
)

# Step 2: Build the sparse constraint matrix that enforces all constrains
M, IFC = build_constraint_matrix(
    phonon, cut_off=15.0, pbc=[False, False, True]
)

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
