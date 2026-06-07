# fMagCalc

A Fortran backend for the linear spin-wave theory (LSWT) calculations of
[`pyMagCalc`](../pyMagCalc). It is the Fortran sibling of `pyMagCalc` (Python)
and `jlMagCalc` (Julia): same physics, different engine.

The goal is to move the **numerical hot path** — per-q diagonalization of the
2N×2N bosonic Bogoliubov Hamiltonian and the S(Q,ω) intensity calculation — out
of Python/NumPy/SciPy into a compiled, OpenMP-parallel Fortran core backed by
LAPACK. `pyMagCalc` keeps its role as the front end: config parsing, symbolic
Hamiltonian construction (SymPy), plotting, and the GUI.

> **`fMagCalc` does not modify `pyMagCalc`.** It lives in its own repository and
> consumes `pyMagCalc` only as a *read-only reference* for validation. Adopting
> the Fortran backend inside `pyMagCalc` later is an opt-in change made *there*,
> not here.

## Why Fortran

The expensive work in LSWT is, for each of potentially many thousands of
q-points:

- evaluating the dynamical matrix `H(q)` (2N×2N, complex),
- a **non-Hermitian** eigen-decomposition (LAPACK `zgeev`) — for S(Q,ω), one at
  `+q` and one at `-q`,
- Bogoliubov normalization (the `α` metric step), degenerate-block
  Gram–Schmidt, +q/−q mode matching, and the K/Kd → intensity contraction.

In `pyMagCalc` this lives in [`magcalc/numerical.py`](../pyMagCalc/magcalc/numerical.py)
and [`magcalc/linalg.py`](../pyMagCalc/magcalc/linalg.py), parallelized with a
`multiprocessing.Pool`. Per-q Python overhead, the GIL, and process-pool
serialization dominate at scale. A Fortran core with `zgeev` inside an OpenMP
`do` loop removes all of that.

## Status

🟢 **M1 (dispersion) + M2 (S(Q,ω)) parity achieved** on the KFe3J example
(3 spins, 6×6, 48 q-points), all vs pyMagCalc as oracle:

| Quantity      | max error vs pyMagCalc |
|---------------|------------------------|
| dispersion E  | 3e-14                  |
| S(Q,ω) E      | 3e-14                  |
| S(Q,ω) I      | 8e-14 abs / 3e-13 rel  |

The full Bogoliubov `KKdMatrix` pipeline (two-stage sort, degenerate-block QR,
α metric, +q/−q matching+phase) is ported and the intensity contraction
matches.

**Speed (KFe3J / 8000 q-points, parity preserved):** the OpenMP compute kernel
is **18–100× faster** than pyMagCalc's multiprocessing `calculate_sqw` — the win
is eliminating Python per-q overhead (GIL, `multiprocessing`, per-q `lambdify`),
since both call the same LAPACK. Called **in-process via ctypes** (M4, the
default backend — no file round-trip), the full S(Q,ω) end-to-end is **~8×
faster**; with the **Phase-2 model path** (Fortran builds H(q) from its bond
decomposition, so nothing per-q is built in Python) it is **~112× faster**
end-to-end. See [docs/BENCHMARK.md](docs/BENCHMARK.md), [PLAN.md](PLAN.md),
[docs/INTERFACE.md](docs/INTERFACE.md).

```python
import fmagcalc                       # backend == "ctypes" when libfmagcalc is built
r = fmagcalc.run_sqw(h_plus, h_minus, ud, ff, S, q_grid)
r["energies"], r["intensities"]       # (Nq, N) each
```

Try it:
```bash
cmake -S . -B build && cmake --build build -j
python python/export_model.py --model-dir ../pyMagCalc/examples/KFe3J \
    --model-module spin_model --out python/fixtures/kfe3j_disp.npz
python python/export_model.py --task sqw --model-dir ../pyMagCalc/examples/KFe3J \
    --model-module spin_model --out python/fixtures/kfe3j_sqw.npz
ctest --test-dir build --output-on-failure
```

## Layout

```
fMagCalc/
├── PLAN.md              # the implementation plan (read this first)
├── docs/INTERFACE.md    # binary data contract between pyMagCalc and the core
├── src/                 # Fortran sources
├── python/fmagcalc/     # thin Python wrapper + model exporter (oracle bridge)
├── tests/               # numerical parity tests against pyMagCalc
├── examples/
└── CMakeLists.txt
```

## Build (once sources exist)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Toolchain on the dev machine: gfortran 15 (Homebrew GCC), CMake ≥ 3.31, LAPACK
via macOS Accelerate (or OpenBLAS). f2py is available for the Python extension
path.

## License

MIT — see [LICENSE](LICENSE).
