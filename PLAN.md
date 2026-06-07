# fMagCalc — Implementation Plan

This plan describes how to build a Fortran backend for `pyMagCalc`'s LSWT
calculations **without touching the `pyMagCalc` codebase**. It is organized
around a clean, narrow data contract so the two projects evolve independently.

---

## 1. What stays in Python, what moves to Fortran

`pyMagCalc` has two distinct stages. The split below is the foundation of the
whole effort.

### Stays in Python (the "front end" / setup — do NOT port)
- **Config & CIF parsing**, magnetic-structure construction, ground-state
  minimization — `magcalc/config_*.py`, `magcalc/generic_model.py`,
  `magcalc/core.py`.
- **Symbolic Hamiltonian construction** — `magcalc/symbolic.py::gen_HM` builds
  the symbolic 2N×2N dynamical matrix `HMat_sym = 2gH` and the rotation matrix
  `Ud_sym`. This is SymPy-heavy, runs **once** per model (and is cached), and
  has no business in Fortran.
- **Plotting and the GUI**.

### Moves to Fortran (the "hot path" — the port target)
Everything in [`magcalc/numerical.py`](../pyMagCalc/magcalc/numerical.py) and
[`magcalc/linalg.py`](../pyMagCalc/magcalc/linalg.py) that runs *per q-point*:

| Step | pyMagCalc reference | Fortran equivalent |
|------|--------------------|--------------------|
| Eigen-decomp of H(q) | `scipy.linalg.eig` / `eigvals` | LAPACK `zgeev` |
| Dispersion: sort, take upper N | `process_calc_disp` | `magcalc_disp` |
| Bogoliubov sort (+/− blocks) | `_diagonalize_and_sort` | `magcalc_kkd` |
| Degenerate-block Gram–Schmidt | `_apply_gram_schmidt` (QR) | LAPACK `zgeqrf`/`zungqr` |
| α metric normalization | `_calculate_alpha_matrix` | `magcalc_kkd` |
| +q/−q mode matching & phase | `_match_and_reorder_minus_q` | `magcalc_kkd` |
| K, Kd build | `_calculate_K_Kd` | `zgemm` |
| Form factor + polarization → intensity | `process_calc_Sqw` (einsum) | `magcalc_sqw` |
| Parallel q-loop | `multiprocessing.Pool` | OpenMP `do` |

This is where ~all the runtime goes, so it captures ~all the speedup.

---

## 2. The Python ↔ Fortran boundary (data contract)

The hard part is feeding `H(q)` to Fortran. Two strategies; we do them in order.

### Phase 1 — **Matrix-stack handoff** (low risk, do this first)
Python evaluates the dynamical matrix on the whole q-grid in one vectorized
shot (lambdify `HMat_sym` with `modules="numpy"`, broadcast over q) and hands
Fortran a dense stack:

- `H_plus`  : complex128, shape `(Nq, 2N, 2N)`  — H(+q)
- `H_minus` : complex128, shape `(Nq, 2N, 2N)`  — H(−q)   *(S(Q,ω) only)*
- `Ud`      : complex128, shape `(3N, 3N)`        — rotation matrix
- `q_grid`  : float64,   shape `(Nq, 3)`
- `ion_ff`  : per-site form-factor coefficients (or precomputed `ff(Nq, N)`)
- scalars: `S` (spin magnitude), `N` (nspins)

Fortran loops over `Nq` with OpenMP, each thread running `zgeev` + the KKd
pipeline, and returns:

- `energies`    : float64, shape `(Nq, N)`
- `intensities` : float64, shape `(Nq, N)`   *(S(Q,ω) only)*

The matrix build stays vectorized in NumPy (fast, one allocation), and the
genuinely expensive serial-per-q linear algebra runs compiled + threaded.
**This already removes the multiprocessing layer and the per-q Python overhead.**

### Phase 2 — **Model export** (max performance, optional)
Eliminate the `(Nq, 2N, 2N)` transfer entirely. An LSWT dynamical matrix has
the closed form

```
H(q) = H0 + Σ_b  M_b · e^{i q·d_b}
```

`gen_HM` already rewrites `exp(±i k·dr)` into sin/cos over a finite set of bond
vectors `dr`. Write a Python **exporter** that walks `HMat_sym` (post-expand)
and collects, per unique bond vector `d_b`, the constant complex coefficient
matrix `M_b`. Fortran then *builds* `H(q)` itself from `{(M_b, d_b)}`, so the
transfer is O(#bonds · (2N)²) instead of O(Nq · (2N)²), and the q-loop has zero
Python involvement. See [docs/INTERFACE.md](docs/INTERFACE.md) for the file
format.

> Start with Phase 1. Only do Phase 2 if profiling shows the matrix transfer /
> NumPy build is a real fraction of runtime.

### Transport mechanism
Pick one (Phase 1 works with any):
- **f2py extension** (recommended): `import fmagcalc._core`, call with NumPy
  arrays, zero file I/O, arrays passed by pointer. Cleanest for the eventual
  `pyMagCalc` opt-in.
- **Standalone exe + binary files**: `fmagcalc-run model.bin → results.bin`.
  Maximum decoupling, language-agnostic, trivial to debug. Good for Phase 2.
- **ctypes / ISO_C_BINDING shared lib**: middle ground.

---

## 3. Fortran module layout (`src/`)

- `magcalc_kinds.f90`   — `dp` (real64), `cp` (complex128) kind params.
- `magcalc_diag.f90`    — `zgeev` wrapper, eigenvalue sort helpers.
- `magcalc_linalg.f90`  — Gram–Schmidt (QR), α metric, +q/−q matching/phase.
- `magcalc_kkd.f90`     — `KKdMatrix` equivalent: orchestrates the Bogoliubov
  pipeline for one q, returns K, Kd, eigenvalues.
- `magcalc_sqw.f90`     — per-q S(Q,ω): form factor + polarization contraction.
- `magcalc_disp.f90`    — dispersion-only path.
- `magcalc_driver.f90`  — OpenMP q-loop; either f2py entry points or `program`
  main reading the binary model.

Match the existing numerical constants exactly (`DEGENERACY_THRESHOLD=1e-8`,
`ZERO_MATRIX_ELEMENT_THRESHOLD=1e-6`, `EIGENVECTOR_MATCHING_THRESHOLD=1e-4`,
`ENERGY_IMAG_PART_THRESHOLD=1e-5`, etc. — see `numerical.py`/`linalg.py`).

---

## 4. Validation strategy (correctness first, speed second)

`pyMagCalc` is the **oracle**. The single most important rule: fMagCalc must
reproduce pyMagCalc's numbers before any optimization claims are made.

1. `python/export_model.py` drives a `pyMagCalc` `MagCalc` instance on a small
   model (e.g. the CVO or KFe3J examples already in `pyMagCalc/cache`), dumps
   the Phase-1 inputs (`H_plus`, `H_minus`, `Ud`, `q_grid`, `S`) **and**
   pyMagCalc's own outputs (`energies`, `intensities`) to an `.npz`.
2. `tests/` runs the Fortran core on those inputs and asserts
   `max|E_fortran − E_py| < 1e-8` and a relative intensity tolerance
   (S(Q,ω) has phase/degeneracy subtleties — match the einsum result, allowing
   for mode-ordering permutations).
3. Use `pyMagCalc`'s existing test fixtures as gold data — they live in
   `pyMagCalc/tests/pckFiles` and `pyMagCalc/cache/symbolic_matrices`.

Watch the subtle parts when porting `linalg.py`:
- the **two-stage sort** in `_diagonalize_and_sort` (split at N, re-sort the
  negative block by `|λ|`),
- the degeneracy-block Gram–Schmidt boundaries,
- the **phase factor** in `_match_and_reorder_minus_q` (ratio of first
  above-threshold components), and the `np.conj(alpha_p · phase)` convention,
- the `Udd_local_boson_map` sign convention (`1/I = -i`).

---

## 5. Milestones

- **M0 — scaffold** ✅ repo, build skeleton, interface doc.
- **M1 — dispersion parity** ✅ Phase-1 handoff via the `fmagcalc_disp` binary
  driver + `zgeev` + sort, matches `process_calc_disp` on the **KFe3J** example
  (3 spins, 6×6, 48 q-points) to **3e-14** (machine precision). Oracle fixture
  exported by `python/export_model.py`; parity checked by
  `tests/test_dispersion_parity.py` (wired into `ctest`).
- **M2 — S(Q,ω) parity** ✅ full `KKdMatrix` pipeline ported
  (`magcalc_kkd.f90`) + intensity contraction (`magcalc_sqw.f90`), driven by
  `fmagcalc_sqw`. Energies match to 3e-14 and **intensities to 8e-14 abs /
  3e-13 rel** on KFe3J. K/Kd themselves differ by the arbitrary per-eigenvector
  phase LAPACK assigns (gauge freedom); the intensity is the gauge-invariant
  bilinear of them and is what the parity test asserts
  (`tests/test_sqw_parity.py`, in `ctest`).
- **M3 — OpenMP + benchmark** ✅ q-loop threaded; on KFe3J/8000 q-points the
  compute-only kernel is **18× faster single-threaded** and **95× at 10 threads**
  than pyMagCalc's multiprocessing run, parity preserved (ΔI ~ 8e-13). See
  [docs/BENCHMARK.md](docs/BENCHMARK.md). (Scaling caps ~4 threads at this tiny
  6×6 size — larger cells will scale further.)
- **M4 — in-process wrapper** ✅ `bind(C)` shared library (`magcalc_capi.f90` →
  `libfmagcalc`) called via ctypes (`python/fmagcalc/_capi.py`); now the default
  backend (`fmagcalc.backend == "ctypes"`), with the subprocess driver as
  fallback. Zero file I/O. Parity tests pass unchanged through it; full S(Q,ω)
  end-to-end is **~6.6× faster than pyMagCalc** (H-build in Python + in-process
  kernel). Chose ctypes over f2py because CMake already links Accelerate+OpenMP
  cleanly. (`run_*` still return plain arrays/dicts; wrapping them in
  `DispersionResult`/`SqwResult` is a thin follow-up if pyMagCalc integration
  (M7) wants it.)
- **M5 — benchmark**: head-to-head timing vs pyMagCalc on a large q-grid;
  document speedup.
- **M6 (optional) — Phase 2 model export** and powder-average loop.
- **M7 (optional, separate PR in pyMagCalc)**: a `backend="fortran"` opt-in flag
  in pyMagCalc that calls this wrapper. Lives in pyMagCalc, gated + falling back
  to the NumPy path.

---

## 6. Non-interference guarantees

- Separate directory, separate git repository (`fMagCalc`, sibling of
  `pyMagCalc`/`jlMagCalc`).
- No edits to any file under `pyMagCalc/`. The exporter imports pyMagCalc as an
  installed/`PYTHONPATH` dependency and only **reads** from it.
- Optional: pin the reference pyMagCalc as a git submodule or record its commit
  hash in test metadata so parity tests are reproducible.
- The eventual integration into pyMagCalc (M7) is additive and opt-in there; the
  default NumPy path is never removed.
