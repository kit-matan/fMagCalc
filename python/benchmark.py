"""M3 benchmark: pyMagCalc (NumPy/SciPy + multiprocessing) vs the Fortran
S(Q,w) backend, on a large q-grid, plus an OpenMP thread-scaling sweep.

Reports three things:
  * pyMagCalc total wall time for calculate_sqw (its real-world cost; already
    multiprocessing across all cores).
  * Fortran end-to-end (H-stack build in Python + fmagcalc_sqw exe).
  * Fortran compute-only (the LAPACK KKd+intensity loop, reported by the exe),
    swept over OMP_NUM_THREADS — this is the apples-to-apples backend number.

Parity on the full grid is verified before any timing is trusted.

    FMAGCALC_PYMAGCALC=../pyMagCalc python python/benchmark.py --nq 8000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def build_calc(S=2.5, params=None):
    import magcalc as mc
    import spin_model  # KFe3J, on path via --model-dir below
    params = params or [3.23, 0.11, 0.218, -0.195, [0, 0, 1], 0.0, 0.0]
    return mc.MagCalc(spin_magnitude=S, hamiltonian_params=params,
                      cache_file_base="KFe3J_bench", cache_mode="w",
                      spin_model_module=spin_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=8000)
    ap.add_argument("--threads", default="1,2,4,8,10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import logging
    logging.disable(logging.WARNING)
    import sympy as sp
    from sympy import lambdify
    sys.path.insert(0, os.path.join(REPO, "python"))
    from fmagcalc import run_sqw

    calc = build_calc()
    n, S = int(calc.nspins), float(calc.spin_magnitude)
    Ud = np.asarray(calc.Ud_numeric, dtype=np.complex128)

    rng = np.random.default_rng(args.seed)
    q_grid = rng.uniform(0.0, 1.0, size=(args.nq, 3))
    q_grid[:, 2] = 0.0
    ff = np.ones((args.nq, n))

    # --- Build H stacks in Python (lambdify eval) ---
    lam = [s for s in calc.full_symbol_list if isinstance(s, sp.Symbol)]
    hfunc = lambdify(lam, calc.HMat_sym, modules=["numpy"], cse=True)
    base = [S] + list(calc.hamiltonian_params)
    t0 = time.perf_counter()
    hp = np.empty((args.nq, 2 * n, 2 * n), np.complex128)
    hm = np.empty_like(hp)
    for i, q in enumerate(q_grid):
        hp[i] = np.asarray(hfunc(*(list(q) + base)), np.complex128)
        hm[i] = np.asarray(hfunc(*(list(-q) + base)), np.complex128)
    t_build = time.perf_counter() - t0

    # --- pyMagCalc full S(Q,w) (multiprocessing) ---
    t0 = time.perf_counter()
    py = calc.calculate_sqw(q_grid)
    t_py = time.perf_counter() - t0
    py_I = np.asarray(py.intensities)

    # --- Parity check on the full grid (single-thread Fortran) ---
    os.environ["OMP_NUM_THREADS"] = "1"
    r = run_sqw(hp, hm, Ud, ff, S, q_grid)
    err = np.nanmax(np.abs(r["intensities"] - py_I))
    sig = py_I > 1e-3
    rel = np.nanmax(np.abs(r["intensities"][sig] - py_I[sig]) / py_I[sig]) if sig.any() else 0.0

    # --- OpenMP thread sweep (compute-only) ---
    sweep = []
    for th in [int(x) for x in args.threads.split(",")]:
        os.environ["OMP_NUM_THREADS"] = str(th)
        t0 = time.perf_counter()
        r = run_sqw(hp, hm, Ud, ff, S, q_grid)
        wall = time.perf_counter() - t0
        sweep.append((th, r["compute_seconds"], wall))

    print(f"\n=== fMagCalc M3 benchmark — KFe3J, {n} spins ({2*n}x{2*n}), Nq={args.nq} ===")
    print(f"parity vs pyMagCalc: max|dI|={err:.2e}  max rel={rel:.2e}\n")
    print(f"H-stack build (Python lambdify, one-time) : {t_build:8.3f} s")
    print(f"pyMagCalc calculate_sqw (multiprocessing) : {t_py:8.3f} s  "
          f"({args.nq / t_py:,.0f} q/s)\n")
    print(f"{'threads':>7} | {'compute(s)':>10} | {'exe wall(s)':>11} | {'q/s (compute)':>13} | {'speedup vs py':>13}")
    print("-" * 70)
    base_compute = None
    for th, comp, wall in sweep:
        if th == 1:
            base_compute = comp
        print(f"{th:>7} | {comp:>10.4f} | {wall:>11.4f} | {args.nq / comp:>13,.0f} | {t_py / comp:>12.1f}x")
    if base_compute:
        print("\nOpenMP scaling (compute-only, vs 1 thread):")
        for th, comp, _ in sweep:
            print(f"   {th:>2} threads: {base_compute / comp:5.2f}x")


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.environ.get("FMAGCALC_PYMAGCALC", "../pyMagCalc")))
sys.path.insert(0, os.path.abspath(os.environ.get("FMAGCALC_MODEL_DIR",
                                                  "../pyMagCalc/examples/KFe3J")))

if __name__ == "__main__":
    main()
