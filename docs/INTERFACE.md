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
```

The metric is `G = diag(1,…,1, −1,…,−1)` (N positive, N negative). The boson map
`Udd` uses `1/I = -i`: rows `3i → (i:+1, i+N:+1)`, rows `3i+1 → (i:-i, i+N:+i)`.
Prefactor for K/Kd is `sqrt(S/2)`.
