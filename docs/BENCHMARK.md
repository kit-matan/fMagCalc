# Benchmark — fMagCalc vs pyMagCalc (M3)

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

→ **~6.6× faster end-to-end**, parity preserved (max |ΔI| = 8e-13). The ratio is
bounded here by the Python H-build (0.55 s); it rises with larger grids (the
build amortizes) and would shrink further with a Phase-2 model export that
builds H(q) in Fortran (PLAN §2).

## Caveat / next step
This is a small-matrix regime that flatters overhead elimination. A fair
follow-up is to benchmark a larger model (more spins) to characterize the
crossover. Thread scaling is overhead-bound at 6×6 and run-to-run noisy beyond
~4 threads; don't over-read the high-thread points.
