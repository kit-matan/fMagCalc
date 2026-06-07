# Benchmark — fMagCalc vs pyMagCalc

## Real-calculation profile (the integration path)

The numbers users actually get: pyMagCalc `calculate_dispersion` /
`calculate_sqw` with `backend="numpy"` vs `backend="fortran"`. The Fortran
backend uses the **exact-H path** (builds H(q) in Python via `lambdify`,
diagonalizes + contracts in Fortran/OpenMP), which is what the GUI/runner call.
All runs OMP across all cores; NumPy uses its `multiprocessing.Pool`.

| Model | Matrix | Nq | Quantity | NumPy | Fortran | Speedup | Parity |
|-------|--------|----|----------|-------|---------|---------|--------|
| KFe3J | 6×6   | 3001 | dispersion | 3.17 s | 0.12 s | **27×** | ΔE 3e-8 |
| KFe3J | 6×6   | 3001 | S(Q,ω)     | 3.58 s | 0.23 s | **16×** | ΔI 2e-12 |
| CVO   | 32×32 | 800  | dispersion | 19.5 s | 2.3 s  | **8.4×** | ΔE 2e-13 |
| CVO   | 32×32 | 800  | S(Q,ω)     | 16.1 s | 2.7 s  | **6.0×** | ΔI 4e-10 |

Reading it:
- **Small cell (6×6):** ~15–35×. `cProfile` of the NumPy path shows ~all the
  time in `multiprocessing.connection.recv` / pool teardown — it is
  **overhead-bound**, not compute-bound, so eliminating the Python per-q
  dispatch dominates the win.
- **Larger cell (32×32):** ~6–8×. Here the LAPACK eigensolve dominates and both
  backends call LAPACK, so the gain narrows to OpenMP threading + removing the
  multiprocessing IPC, minus the shared Python `lambdify` H-build (a real
  fraction of the Fortran end-to-end time).
- Parity is machine-precision-ish throughout (ΔE/ΔI ≤ ~1e-8), so the speedups
  compare equal answers.

Reproduce: `python /tmp/profile_real.py` style — drive `calc.calculate_*(qg,
backend=...)` both ways and compare (see git history of this change).

---

## Micro-benchmark — Fortran kernel paths (M3)

The section below isolates the Fortran kernel throughput (subprocess driver and
the Phase-2 model path), separate from the end-to-end integration above. The
model path (`run_sqw_model`) reaches ~112× but rebuilds H(q) from a bond
decomposition; it is **not** the default integration path because that
reconstruction is fragile at degenerate modes (see docs/INTERFACE.md).

Reproduce with:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
FMAGCALC_PYMAGCALC=../pyMagCalc python python/benchmark.py --nq 8000
```

## Setup
- Model: **KFe3J** example (3 spins → 6×6 bosonic Hamiltonian), the same model
  used for the M1/M2 parity tests.
- Grid: 8000 random in-plane q-points.
- Machine: Apple Silicon, 10 cores; gfortran 15 (`-O3 -march=native`), LAPACK
  via Accelerate; pyMagCalc on CPython 3.12 using `multiprocessing.Pool` across
  all cores.
- **Parity is verified on the full 8000-point grid before timing is reported**
  (max |ΔI| = 8.1e-13, max relative = 2.3e-12). Speed claims are only
  meaningful because the answers match.

## Results (Nq = 8000)

| stage | time | throughput |
|-------|------|------------|
| H-stack build (Python `lambdify` eval, one-time) | 0.90 s | — |
| pyMagCalc `calculate_sqw` (multiprocessing, all cores) | 4.73 s | 1,692 q/s |

| Fortran threads | compute-only | q/s (compute) | speedup vs pyMagCalc |
|----------------:|-------------:|--------------:|---------------------:|
| 1  | 0.262 s | 30,545  | 18.1× |
| 2  | 0.130 s | 61,333  | 36.2× |
| 4  | 0.067 s | 119,980 | 70.9× |
| 8  | 0.058 s | 138,480 | 81.8× |
| 10 | 0.050 s | 161,323 | 95.3× |

**OpenMP scaling (compute-only, vs 1 thread):** 2.01× / 3.93× / 4.53× / 5.28×
at 2 / 4 / 8 / 10 threads.

## Reading the numbers honestly

- **Single-thread Fortran is already ~18× faster** than pyMagCalc running on all
  cores. That gap is *not* better linear algebra — both call the same LAPACK
  `zgeev`. It is the elimination of Python per-q overhead: the GIL, the
  `multiprocessing` pickle/dispatch, and a per-q SymPy-`lambdify` evaluation
  inside every worker. fMagCalc moves that loop into compiled code.
- **End-to-end**, the Fortran path still needs the H-stack built in Python
  (0.90 s here, shared cost). So a full user-facing S(Q,ω) run is
  ~0.90 + 0.05 ≈ 0.95 s vs pyMagCalc's 4.73 s → **~5× end-to-end** at this size,
  rising toward the compute-only ratio as the grid grows (the 0.90 s build
  amortizes).
- **Scaling caps early** because a 6×6 diagonalization is tiny: per-q work is
  small relative to threading/memory overhead, so beyond ~4 threads there's
  little left to parallelize. Larger magnetic cells (bigger 2N×2N matrices)
  will (a) scale further and (b) shift more of the total into LAPACK itself,
  where the compiled advantage is smaller but the overhead win persists.
- The Fortran "exe wall" column (in the script output) exceeds compute-only
  because it includes reading the H stacks and writing the K/Kd/evals debug
  arrays. `compute_seconds` (measured around the kernel inside the driver)
  excludes all file I/O and is the right number for the scaling study. The
  file round-trip itself disappears with the in-process f2py path (M4).

## In-process path (M4)
The numbers above time the standalone `fmagcalc_sqw` exe (subprocess + binary
files). The default backend is now the **in-process ctypes path** (libfmagcalc
shared library, no file I/O). End-to-end for the full S(Q,ω) at Nq=8000:

| step | time |
|------|------|
| H-stack build (Python `lambdify`, one-time) | 0.55 s |
| Fortran in-process kernel (ctypes, all threads) | 0.14 s |
| **fMagCalc end-to-end** | **0.68 s** |
| pyMagCalc `calculate_sqw` | 4.48 s |

→ **~8× faster end-to-end** (run-to-run 6–8×), parity preserved (max |ΔI| =
8e-13). The ratio is bounded here by the Python H-build (0.55 s).

## Phase-2 model path (build H(q) in Fortran)
The Python H-build is removed entirely: the dynamical matrix is exported once as
its bond decomposition `H(q) = Σ_b M_b e^{i q·d_b}` (13 bonds for KFe3J) and
Fortran reconstructs H(±q) inside the q-loop. Nothing per-q crosses from Python.

| path | end-to-end (Nq=8000) | speedup | parity |
|------|----------------------|---------|--------|
| pyMagCalc | 5.71 s | 1× | — |
| H-stack (M4) | 0.68 s | ~8× | 8e-13 |
| **model (Phase 2)** | **0.051 s** | **~112×** | 2e-10 |

The model path runs at ~157,000 q/s. Bond extraction is a one-time 0.54 s
(parameter-set-specific, reusable across any q-grid). Its parity floor (~2e-10)
is slightly looser than the H-stack path because the bond coefficients come
through a symbolic rewrite/expand — still far inside physical tolerance.

## Caveat / next step
This is a small-matrix regime that flatters overhead elimination. A fair
follow-up is to benchmark a larger model (more spins) to characterize the
crossover. Thread scaling is overhead-bound at 6×6 and run-to-run noisy beyond
~4 threads; don't over-read the high-thread points.
