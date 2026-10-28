# Python ↔ Fortran data contract

All arrays are **column-major** on the Fortran side. When passing NumPy arrays
through f2py, declare them `order='F'` (or let f2py copy) to avoid silent
transposes. Complex is IEEE complex128 (`complex(kind=cp)`), real is
float64 (`real(kind=dp)`).

Dimensions: `N` = number of spins in the magnetic unit cell, `Nq` = number of
q-points. The bosonic Hamiltonian is `2N × 2N`; the rotation matrix is
`3N × 3N`; K/Kd are `3N × 2N`.

## Phase 1 — matrix-stack handoff

### Inputs (Python → Fortran)
| Name      | Type/shape                 | Meaning |
|-----------|----------------------------|---------|
| `S`       | real scalar                | spin magnitude |
| `spins`   | real `(N)`, optional       | per-site S_i of the `sqrt(S_i/2)` intensity prefactor (mixed spin); default N copies of `S`. ctypes path only — the binary file below carries one `S` |
| `N`       | int scalar                 | nspins |
| `q_grid`  | real `(Nq, 3)`             | q-vectors (the path or grid) |
| `H_plus`  | complex `(Nq, 2N, 2N)`     | dynamical matrix at +q (= 2gH) |
| `H_minus` | complex `(Nq, 2N, 2N)`     | dynamical matrix at −q (S(Q,ω) only) |
| `Ud`      | complex `(3N, 3N)`         | local→global rotation matrix |
| `ff`      | real `(Nq, N)`             | magnetic form factor per site per q (precomputed in Python) |

### Outputs (Fortran → Python)
| Name          | Type/shape       | Meaning |
|---------------|------------------|---------|
| `energies`    | real `(Nq, N)`   | magnon energies (upper-half eigenvalues) |
| `intensities` | real `(Nq, N)`   | S(Q,ω) intensity per mode (S(Q,ω) only) |
| `info`        | int `(Nq)`       | 0 = ok, nonzero = zgeev/KKd failure at that q (→ NaN) |

Form factors are computed in Python (`magcalc/form_factors.py`) so the Fortran
core stays free of the ion lookup tables. A q with any failure returns NaN for
that row, mirroring pyMagCalc's `nan_energies`/`nan_result` behavior.

## Phase 2 — model export (optional)

Replaces the `(Nq, 2N, 2N)` stacks with a closed-form description so Fortran
builds `H(q)` itself: `H(q) = Σ_b M_b · exp(i q·d_b)`.

### `model.bin` layout (little-endian)
```
magic       : char[8]  = "FMAGCAL1"
N           : int32                      # nspins
S           : float64
n_bonds     : int32
Ud          : complex128[3N, 3N]         # column-major
for each bond b in 0..n_bonds-1:
    d_b     : float64[3]                 # bond vector
    M_b     : complex128[2N, 2N]         # coefficient matrix (column-major)
```
Fortran reconstructs, per q: `H(q) = Σ_b M_b * cexp(i * dot(q, d_b))`. The −q
matrix is obtained by negating q. The exporter
(`python/export_model.py`) is responsible for collecting `{(M_b, d_b)}` from
`HMat_sym` after `gen_HM`.

## Numerical constants (must match pyMagCalc exactly)

From `magcalc/linalg.py` and `magcalc/numerical.py`:
```
DEGENERACY_THRESHOLD              = 1e-8
ZERO_MATRIX_ELEMENT_THRESHOLD     = 1e-6
ALPHA_MATRIX_ZERO_NORM_WARNING    = 1e-14
EIGENVECTOR_MATCHING_THRESHOLD    = 1e-4
ENERGY_IMAG_PART_THRESHOLD        = 1e-5
SQW_IMAG_PART_THRESHOLD           = 1e-4
Q_ZERO_THRESHOLD                  = 1e-10
MODEL_ENERGY_ZERO_THRESHOLD       = 1e-3   # fMagCalc-only, bond-model path only (see below)
```

`MODEL_ENERGY_ZERO_THRESHOLD` has no pyMagCalc counterpart, and applies only to
`run_sqw_model`, the path that rebuilds H(q) from the bond list. That rebuild differs
from the exact H by ~1e-14, which is enough to give a zero-energy Goldstone mode a
spurious `~3S` intensity: its Bogoliubov normalization is ill-conditioned (a null
vector of the para-unitary metric), so the result depends on the arbitrary
degenerate eigenbasis. Without the cut that shows up as vertical streaks in S(Q,ω);
`tests/test_streak_regression.py` pins it.

`run_sqw`, the exact-H path pyMagCalc calls, applies **no** energy cut (2026-10-28).
There the pseudo-norm rule shared with pyMagCalc (alpha = 0 when |N_ii| <
`ZERO_MATRIX_ELEMENT_THRESHOLD`²) already zeroes a genuine zero mode. An energy cut
also drops well-conditioned modes: on the ferromagnetic chain, whose intensity is
(S/2)(1 + Q̂_z²) exactly at every q, the old 1e-3 meV cut zeroed it at every mode
from 4e-11 to 9.9e-4 meV. The two backends' S(Q,ω) and powder averages therefore
now agree on gapless models too (`tests/test_degenerate_and_gapless.py`).

The metric is `G = diag(1,…,1, −1,…,−1)` (N positive, N negative). The boson map
`Udd` uses `1/I = -i`: rows `3i → (i:+1, i+N:+1)`, rows `3i+1 → (i:-i, i+N:+i)`.
Prefactor for K/Kd is `sqrt(S_i/2)` **per site** (rows `3i..3i+2` of site i), as in
pyMagCalc's `linalg._spin_prefactor_vector`; with every S_i = S it is the old
`sqrt(S/2)` element for element, so a uniform model is bit-identical to the scalar
kernel. In the C API (`magcalc_sqw_c`, `magcalc_sqw_model_c`) the third argument is a
pointer to those N doubles; it was the scalar `S` by value before 2026-10-27 (K6).
