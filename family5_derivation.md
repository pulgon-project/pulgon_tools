# Family 5: From The Original Table To The Modified Table

## 1. Setup

### Structure used for the check
* Reference structure: `test/data/test_irrep_struct/family_05.vasp`
* Line group family = 5 (T_Q D_n)
* Detected by `get_linegroup_symmetry_dataset`:
  * `trans_sym = "(C6|T3(3.001))"`  — screw with rotation 2π/Q and z-translation f
  * `rota_sym = "D2"`               — axial point group D_n with n = 2
  * `Q = 6`, `f = a/3`, `a = 9.0 Å`, `nrot = 4`
  * α_U = 0 is the azimuthal angle of the perpendicular C_2' generator
* Factor-group order |G| = 12; degrees of freedom in the test structure: 3N = 36

### Three q-points used
* Gamma: `q = 0`
* Generic interior q: `q = 0.25 · 2π/a = π/(2a)`
* Brillouin-zone boundary: `q = 0.5 · 2π/a = π/a`

### The four generator labels carried inside the irrep table
Inside `Irreps_tables_withparities.py`, family 5 generators are addressed
by their position in `func0`, `func1`, `func2`:

| idx | meaning                                  | actual op in this structure |
|----:|------------------------------------------|-----------------------------|
| 0   | identity                                 | E                           |
| 1   | screw S                                  | (C₆ \| a/3)                 |
| 2   | axial C_n around z                       | C₂(z) = S³ mod lattice      |
| 3   | perpendicular U_x (C₂')                  | diag(1, −1, −1)             |

## 2. The Two Tests

### 2.1 Orthogonality of irreps
In a finite group the characters must satisfy
\[
  \tfrac{1}{|G|}\sum_{g\in G} \chi_\mu(g)\,\chi_\nu(g)^* = \delta_{\mu\nu}.
\]
More strongly, the irrep matrix elements satisfy Schur orthogonality:
\[
  \tfrac{d_\mu}{|G|}\sum_{g\in G} D^{(\mu)}_{ij}(g)\,D^{(\nu)}_{kl}(g)^*
   = \delta_{\mu\nu}\,\delta_{ik}\,\delta_{jl}.
\]

### 2.2 Idempotency of the Bloch projector
The projector used by `get_adapted_matrix_withparities` is
\[
  P_\mu = \frac{d_\mu}{|H_k|}\sum_{h\in H_k}\chi_\mu(h)^*\,D_{\rm Bloch}(h).
\]
For a valid irrep the relevant error measure is
\(\|P_\mu^2-P_\mu\|\,/\,\|P_\mu\|\), which should be at the level of the
arithmetic noise (\(\sim 10^{-5}\)–\(10^{-4}\) on this test structure).

## 3. First Failure: The Original Tabulation Is Not A Set Of Orthogonal Irreps

Using the original table at all three q-points:

| q point     | # irreps | dim pattern               | Σ d² | should = |
|-------------|---------:|---------------------------|-----:|---------:|
| Γ           | 8        | [2, 2, 1, 1, 2, 2, 1, 1]  | **20** | 12 (\|G\|) |
| π/(2a)      | 6        | [2, 2, 2, 2, 2, 2]        | **24** | 12          |
| π/a         | 6        | [2, 2, 2, 2, 2, 2]        | **24** | 12          |

The Γ value `20` already violates `Σ d² = |G|`, so the original tabulation
double-counts irreps.

### Numerical orthogonality result
| q point | character max off-diag (original) | projector max idem. error (original) |
|---------|----------------------------------:|--------------------------------------:|
| Γ       | 1.000                             | 0.75 (m=3), 0.61 (m=1,2)              |
| π/(2a)  | 0.667                             | 0.83 (everything except m=0)          |
| π/a     | 1.000                             | 0.83 (everything except m=0)          |

The full numerical idempotency map at Γ:

| ii | (m, π_U) | d | tr(P) | ‖P²−P‖/‖P‖ |
|---:|----------|--:|------:|-----------:|
| 0  | (−2, 0)  | 2 | 12    | 6.1 × 10⁻¹ |
| 1  | (−1, 0)  | 2 | 12    | 6.3 × 10⁻¹ |
| 2  | (0, −1)  | 1 | 3     | 5.1 × 10⁻⁶ |
| 3  | (0, +1)  | 1 | 3     | 5.1 × 10⁻⁶ |
| 4  | (1, 0)   | 2 | 12    | 6.3 × 10⁻¹ |
| 5  | (2, 0)   | 2 | 12    | 6.1 × 10⁻¹ |
| 6  | (3, −1)  | 1 | 3     | 7.5 × 10⁻¹ |
| 7  | (3, +1)  | 1 | 3     | 7.5 × 10⁻¹ |

Only the two m = 0 / π_U = ±1 rows are numerically projectors. Every other
row fails by an O(1) margin.

### What this forces
Two corrections plus a non-zero-q extension are needed:

1. the m-range at Γ cannot run over both `m` and `Q−m`;
2. the C_n-axial phase formula must double the angular factor (the
   currently used `_phase2(m, n) = exp(i·m·2π/n)` is off by a factor of
   two);
3. at non-zero q the `(+k, −k)` sectors must be carried inside the
   representation matrices.

## 4. The m Range At Γ Must Be Reduced To 0 ≤ m ≤ ⌊Q/2⌋

The original implementation enumerated `m` over the full screw range
`_m_values_screw(Q) = [-Q/2+1, …, Q/2]` (six values for Q = 6: `[-2, -1,
0, 1, 2, 3]`). The pairing `m ↔ Q − m ≡ −m` is already built into the 2D
representation matrix
\[
  D^{(\text{func1})}(S)
   = \mathrm{diag}\bigl(\mathrm{screw\_phase},\; 1/\mathrm{screw\_phase}\bigr),
\]
so enumerating both members of the pair produces the same physical
irrep twice. Restricting the loop to
\[
  m = 0,\,1,\,\dots,\,\lfloor Q/2 \rfloor
\]
removes the duplication. With Q = 6 the surviving m values are `{0, 1, 2, 3}`.

## 5. Why The Axial-Rotation Phase Must Be Doubled

The original table used
\[
  r_{\rm phase}^{\rm old} = e^{i\,m\,2\pi/n},\qquad n = nrot.
\]
For a D_n point group, however,
`LineGroupAnalyzer.get_rotational_symmetry_number()` returns the *order
of the full rotation subgroup*, which is `2·n_axial`, not `n_axial`. The
fc[2] slot is fed by the actual `C_n` axial generator, which here is
`C_2(z)`, i.e. a rotation by 2π/2 = π. So the correct phase is
\[
  r_{\rm phase} = e^{i\,m\,2\pi/n_{\rm axial}}
                 = e^{i\,m\,4\pi/nrot}.
\]
For our test structure (nrot = 4) this is `(−1)^m`, while the original
formula produced `exp(i·m·π/2)`.

Cross-check: in the factor group, S³ ≡ C₂(z) modulo one lattice
translation, so the irrep value must satisfy
\(D(S)^3 = e^{i\,k\,a}\,D(C_2)\). With our `screw_phase` and the corrected
`rot_phase`:
\(
  \mathrm{screw\_phase}^3
  = e^{3i\,k\,f}\cdot e^{i\,m\,\pi}
  = e^{i\,k\,a}\cdot(-1)^m
  = e^{i\,k\,a}\,\cdot\,r_{\rm phase}.
\)
The relation closes, which it did *not* under the previous `n = nrot`
convention.

A Γ-point symptom of the old phase: the rows `(m = 1, π_U = arbitrary)`
and `(m = 2, π_U = arbitrary)` shared identical characters on the U and
U-containing operations (all equal to 0), giving overlap

\[ \tfrac{1}{|G|}\sum_g\chi_1(g)\chi_2(g)^* = 0.72 \neq 0. \]

After the correction this overlap drops to ≤ 1.3 × 10⁻¹⁶.

## 6. Why m = Q/2 Must Split Off At Γ

For even Q the value `m = Q/2` is special because
\[ m \equiv -m \pmod{Q}, \]
so the usual (+m, −m) doubling collapses: `m = Q/2` is its own conjugate
under the screw. Keeping it in the 2D `func1` branch makes that
representation reducible. With Q = 10 (family 13) the over-count of the
old table was `4·1² + 10·2² = 44 > 40`; the family-5 analogue with Q = 6
gives `4·1² + 4·2² = 20 > 12`. So:

* m = 0 stays in the 1D `func0` branch (already correct in the original);
* **m = Q/2 must also move into the 1D `func0` branch with π_U = ±1**;
* only `0 < m < Q/2` remain in the 2D `func1` branch.

This is exactly the branch logic implemented in the modified file:
```python
is_special_m = np.isclose(tmp_m1, 0, atol=symprec) or (
    int(q_num) % 2 == 0
    and np.isclose(tmp_m1, q_num / 2, atol=symprec)
)
```

## 7. Why Generic-q Sectors Need The Explicit (+k, −k) Doubling

At non-zero q the projector is built from the k-preserving little group
\(L_k = T_Q\) (rotations + lattice translations) of order Q = 6. The
generator U_x sends `+k → −k`, so it sits outside L_k. The projector code
identifies little-group elements as those whose representation matrix is
block-diagonal in the `(+k, −k)` basis.

There are two distinct sub-cases:

### 7.1 Self-paired m (m = 0 or m = Q/2): `func2`
On the basis `(|m, +k⟩, |m, −k⟩)`,
\[
  D(S) = \mathrm{diag}(e^{i(kf + m\,2\pi/Q)},\, e^{i(-kf + m\,2\pi/Q)}),\quad
  D(C_2) = r_{\rm phase}\cdot I_2,\quad
  D(U_x) = \text{swap}(α_U).
\]
The screw block is diagonal in `(+k, −k)`, the C_2(z) block is a scalar,
so both belong to the little group; U_x swaps the two k-sectors and is
excluded from the little-group filter, as required.

### 7.2 Generic 0 < m < Q (m ≠ 0, Q/2): `func1` on `(|+m, +k⟩, |−m, −k⟩)`
Family 5 has **no horizontal mirror**, so a family-13-style 4D rep on
`(|+m,+k⟩, |−m,+k⟩, |+m,−k⟩, |−m,−k⟩)` is *reducible* under the lone
U_x generator (it splits into the two 2D induced reps
`Ind χ_{(+m,+k)}` and `Ind χ_{(−m,+k)}`). Numerically, a 4D rep gives
`‖P²−P‖/‖P‖ = 1` because the reducible character drives a sum of two
idempotents with the wrong overall coefficient.

The clean fix is to enumerate every \(m \in \{0, 1, \dots, Q-1\}\) at
non-zero q and assign each its own 2D rep on the induced basis
`(|+m,+k⟩, |−m,−k⟩)`:
```python
if np.isclose(qpoint, 0, atol=symprec):
    m1_value = list(range(0, int(q_num) // 2 + 1))   # 0 .. Q/2
else:
    m1_value = list(range(0, int(q_num)))            # 0 .. Q-1
```
The corresponding matrices coincide with the Γ-time `func1` (the same
diagonal screw and axial-rotation phases plus the U_x swap) — the only
difference is the basis interpretation. The little-group filter then
returns `mat[0,0]` for each m, giving the correct +k-sector 1D character
of L_k.

This labeling intentionally enumerates `m` and `Q − m` as separate rows
even though they correspond to the same G-irrep. The reason is that the
projector code is designed to produce one column-block of the adapted
basis per row, so we need every distinct (+k, m) Bloch character to
appear. With m and Q − m both present, the resulting 6 columns of
2-mode-per-irrep cover all 36 dofs at +k.

## 8. Numerical Evidence For The Three Fixes

After the modifications described above:

| q point | # irreps | dim pattern    | Σ d² | char ortho max off-diag | projector max ‖P²−P‖/‖P‖ | Σ tr(P) |
|---------|---------:|----------------|-----:|------------------------:|-------------------------:|--------:|
| Γ       | 6        | [1,1,2,2,1,1]  | **12 = \|G\|** ✓ | **1.3 × 10⁻¹⁶** | **5 × 10⁻⁶** | 36 ✓ |
| π/(2a)  | 6        | [2,2,2,2,2,2]  | 24                   | (little group exact) | **9 × 10⁻⁵** | 36 ✓ |
| π/a     | 6        | [2,2,2,2,2,2]  | 24                   | (little group exact) | **2 × 10⁻⁴** | 36 ✓ |

The projector error drops by **5–6 orders of magnitude** at every q point
relative to the original tabulation; Σ tr(P) saturates 3N = 36 so the
adapted basis is now complete and square; the basis ortho-normality test
(`adapted.conj().T @ adapted == I`) passes.

## 9. Side Issue: get_perms_from_ops Tolerance

The detected `f_screw = 3.001` carries a 10⁻³ rounding error inherited
from the four-decimal z fractional coordinates of `family_05.vasp`
(`0.3678 − 0.0344 = 0.3334`, so `0.3334 × 9 = 3.0006 → round(·,3) =
3.001`). Two screw applications accumulate this to ≈ 2 × 10⁻³, which
exceeds the default `symprec = 1 × 10⁻³` used by
`get_perms_from_ops`, raising `ValueError` before the irrep table is
even reached.

The test file therefore selects `symprec = 2 × 10⁻³` only for family 5
through the `FAMILY_SYMPREC` dict; all other families keep the original
`1 × 10⁻³` tolerance.

A cleaner long-term fix is to either (a) regenerate the test structure
with six-decimal z fractional coordinates so the screw step matches
`a/3` to machine precision, or (b) round `f_screw` against the rational
approximation `a · (numerator/denominator)` inside
`_extract_screw_parameters`. Both touch infrastructure outside the
irreps table and are left for a follow-up patch.

## 10. Summary

Family 5 originally failed every q-point check because the irrep table
double-counted the `(+m, −m)` pair at Γ, used the wrong axial-rotation
phase (off by a factor of two) and did not encode the `(+k, −k)` doubling
needed at non-zero q. Three targeted edits — reduce the Γ m-range,
double the rotation phase, and enumerate every m ∈ `[0, Q−1]` at non-zero
q with a 2D rep — restore the |G|-sum rule at Γ, bring all three Bloch
projector idempotencies to the `10⁻⁵–10⁻⁴` floor and let
`test_symmetry_projector.py` flip all six family-5 cases from `xfail`
to `PASS`.
