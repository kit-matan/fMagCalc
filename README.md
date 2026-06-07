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

🚧 **Scaffold only.** See [PLAN.md](PLAN.md) for the full implementation plan and
[docs/INTERFACE.md](docs/INTERFACE.md) for the Python↔Fortran data contract. The
files in `src/` are stubs that establish the module layout and build.

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
